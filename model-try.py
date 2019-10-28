from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

from transformers import ALBertConfig, ALBertForPreTraining

import logging
logging.basicConfig(level=logging.INFO)
PATH = "./pytorch_albert"

config = ALBertConfig.from_json_file("./albert_config.json")
model = ALBertForPreTraining(config)
model.load_state_dict(torch.load(PATH))
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param)