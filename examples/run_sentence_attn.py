import os
import argparse
import logging
from datetime import timedelta, datetime
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import (WEIGHTS_NAME,
                                  BertConfig, BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer)

from run_squad_albert import set_seed, load_and_cache_examples, MODEL_CLASSES

from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad
from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)
from run_squad import load_and_cache_examples, to_list

from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 12), dpi=160, facecolor='w', edgecolor='k')

logger = logging.getLogger(__name__)

def save_attention_plot(args, unique_id, head_id, attention, doc_tokens):
    attention = attention[:len(doc_tokens), :len(doc_tokens)]
    plt.imshow(attention.detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.xticks(range(len(doc_tokens)), doc_tokens, rotation=90, fontsize='xx-small')
    plt.yticks(range(len(doc_tokens)), doc_tokens, rotation=0, fontsize='xx-small')
    plt.colorbar()
    file_name = os.path.join(args.output_dir, '{}_{}.jpg'.format(unique_id, head_id))
    plt.savefig(file_name)
    plt.clf()

def plot_attention(args, feature, all_attentions, heads=None):
    # heads: to specify which head to visualize.
    # format: list of tuples => [(1, 3), (2, 7), (3, 9), ...]
    # (3, 9) means layer 3 head 9. (layer/head index starts from 1)
    doc_tokens = feature.tokens
    unique_id = feature.unique_id
    if heads is not None:
        for index in heads:
            layer = index[0]
            head = index[1]
            attention = all_attentions[layer][head]
            head_id = '{}_{}'.format(layer, head)
            save_attention_plot(args, unique_id, head_id, attention, doc_tokens)
    else:
        # all attention: list (len = num_layers) of attentions (shape = [num_head, seq, seq])
        for i, layer_attention in enumerate(all_attentions):
            # layer_attention : [num_head, seq, seq]
            for j, attention in enumerate(layer_attention):
                # attention : [seq, seq]
                head_id = '{}_{}'.format(i, j)
                save_attention_plot(args, unique_id, head_id, attention, doc_tokens)
        

def compute_self_attention(args, model, eval_dataloader, eval_examples, eval_features, head_mask=None):
    n_layers, n_heads = model.bert.config.num_hidden_layers, model.bert.config.num_attention_heads

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)
    all_result = []

    for step, batch in enumerate(tqdm(eval_dataloader, desc='Iteration', disable=args.local_rank not in [-1, 0])):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, start_positions, end_positions, cls_index, p_mask, example_indices = batch
        inputs = {  'input_ids': input_ids,
                    'attention_mask': input_mask,
                    'token_type_ids': None if args.model_type == 'xlm' else segment_ids,
                    'head_mask': head_mask}
        if args.model_type in ['xlnet', 'xlm']:
            inputs.update({'cls_index': cls_index, 'p_mask': p_mask})
        
        outputs = model(**inputs)
        #start_logits, end_logits, all_attentions = outputs
        all_attentions = outputs[-1]
        # all_attentions : list (len = num_layers) of attentions (each shape = [batch_size, num_head, seq, seq])
        for i, example_index in enumerate(example_indices):
            feature = eval_features[example_index.item()]
            unique_id = int(feature.unique_id)
            attention = [x[i, :, :] for x in all_attentions]
            if args.model_type in ['xlnet', 'xlm']:
                result = RawResultExtended(unique_id           = unique_id,
                                           start_top_log_probs = to_list(outputs[0][i]),
                                           start_top_index     = to_list(outputs[1][i]),
                                           end_top_log_probs   = to_list(outputs[2][i]),
                                           end_top_index       = to_list(outputs[3][i]),
                                           cls_logits          = to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id    = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits   = to_list(outputs[1][i]))
            all_result.append(result)
            plot_attention(args, feature, attention)

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="The input data file.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list:" + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Whether to overwrite data in output directory")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, sequences shorter padded.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
    
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument('--subset', default=10, type=int,
                        help='If > 0: limit the data to a subset of data_subset instances.')

    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    args = parser.parse_args()

    # Setup devices and distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')  # Initializes the distributed backend

    # Setup logging
    logging.basicConfig(level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed: {}".format(args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seeds
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    '''
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, output_attentions=True)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    '''
    config = config = config_class.from_json_file("config/albert_config.json")
    config.output_attentions = True
    assert config.output_attentions == True
    tokenizer = tokenizer_class(vocab_file="spm_model/30k-clean.model")
    tokenizer.do_lower_case = args.do_lower_case
    

    eval_data, eval_examples, eval_features = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=True)
    # we need to add all_input_ids in order to make evaluation feasible.
    tensors = eval_data.tensors
    all_input_ids = tensors[0]
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    tensors = tensors + (all_example_index,)
    eval_data = TensorDataset(*tensors)
    if args.subset > 0:
        eval_data = Subset(eval_data, list(range(min(args.subset, len(eval_data)))))

    eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    # the structure should be like this (if args.output_dir == SQuAD_models):
    # SQuAD_models
    # ├─ checkpoint_2000
    # ├─ checkpoint_4000
    # └─ checkpoint_6000
    output_dir = args.output_dir

    model_list = os.listdir(args.model_name_or_path)
    for model_path in model_list:
        model_path = os.path.join(args.model_name_or_path, model_path)
        if os.path.isdir(model_path):
            model = model_class.from_pretrained(model_path, from_tf=bool('.ckpt' in model_path), config=config)
            if args.local_rank == 0:
                torch.distributed.barrier()
            model.to(args.device)
            if args.local_rank != -1:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                                  output_device=args.local_rank,
                                                                  find_unused_parameters=True)
            elif args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            args.output_dir = os.path.join(output_dir, model_path)
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            #DO STH.
            compute_self_attention(args, model, eval_dataloader, eval_examples, eval_features)

if __name__ == '__main__':
    main()
