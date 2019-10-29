from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

from transformers import ALBertConfig, ALBertForPreTraining,ALBertForQuestionAnswering
from transformers import ALbertTokenizer
import logging
logging.basicConfig(level=logging.INFO)
def INFO(model,tokenizer):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
    ids = tokenizer.encode(["albert"])
    token=tokenizer.decode(ids)
    print(ids)
    print(token)
    print("well-done!!!")
    return 

if __name__ == "__main__":
    PATH = "./pytorch_albert"
    Spm_model = "30k-clean.model"
    config = ALBertConfig.from_json_file("./albert_config.json")
    model = ALBertForPreTraining(config)
    model_qa = ALBertForQuestionAnswering(config)
    model_qa.load_state_dict(torch.load(PATH),strict=False)
    tokenizer = ALbertTokenizer(vocab_file=Spm_model)
    # INFO(model,tokenizer)

