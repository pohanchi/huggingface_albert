from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

from transformers import ALBertConfig, ALBertForPreTraining
from transformers import ALbertTokenizer
import logging
logging.basicConfig(level=logging.INFO)
PATH = "./pytorch_albert"
Spm_model = "30k-clean.model"
config = ALBertConfig.from_json_file("./albert_config.json")
model = ALBertForPreTraining(config)
model.load_state_dict(torch.load(PATH))
tokenizer = ALbertTokenizer(vocab_file=Spm_model)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param)

ids = tokenizer.encode(["albert"])
token=tokenizer.decode(ids)
print("well-done!!!")
print(ids)
print(token)