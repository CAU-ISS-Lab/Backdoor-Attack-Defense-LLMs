#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import argparse
import dataclasses
import json
import logging
import os
import time
from pathlib import Path
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric
import sys

from transformers import (
    AutoModelForSequenceClassification,
)
from transformers import AutoTokenizer,set_seed

def get_models(path_models, num_labels,seed):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(path_models, output_hidden_states = True,num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(path_models)
    return model,tokenizer


def get_path(path):
    path2 = path

    with open(path2, 'r', encoding='utf-8') as file:
        data = json.load(file)
    path_models = data['best_model_checkpoint'].replace("\\", "/")
    path_results = path_models.replace("victims",'defense')
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    path_results += "/results.txt"
    num_labels = 2
    if "agnews" in path_models:
        num_labels = 4
    return path_models,path_results,num_labels

