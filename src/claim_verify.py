# a copy from run_classfier to test tokenizer


# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import pickle
import argparse
import csv
import logging
import os
import random
import sys
import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler, ConcatDataset,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef,accuracy_score, f1_score

import transformers
from transformers import RobertaForSequenceClassification, RobertaConfig, BertTokenizer
from transformers import AutoModel, AutoTokenizer
from transformers import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW, get_linear_schedule_with_warmup

from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
# from transformers.modeling_utils import *
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification


# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
# from pytorch_pretrained_bert.modeling import BertConfig
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

logger = logging.getLogger(__name__)


# INPUT EXAMPLE
class InputExample_train(object):
    """A single training example for simple sequence classification."""

    def __init__(self, 
                 guid,
                 evidence,
                 claim,
                 label,
                 domain):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            evidence (string): The 
            claim: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.evidence = evidence
        self.claim = claim
        self.label = label
        self.domain = domain

class InputExample_dev(object):
    """A single developer example for simple sequence classification."""

    def __init__(self, 
                 guid,
                 claim_id,
                 num_sentence,
                 evidence,
                 claim,
                 domain,
                 label):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            evidence (string): The 
            claim: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.claim_id = claim_id
        self.num_sentence = num_sentence
        self.evidence = evidence
        self.claim = claim
        self.domain = domain
        self.label = label

class InputFeatures_train(object):
    """A single set of features of data."""

    def __init__(self, input_ids,
                input_mask,
                segment_ids,
                label_id,
                guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
    
        self.label_id = label_id
        self.guid = guid

class InputFeatures_dev(object):
    """A single set of features of data."""

    def __init__(self, input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 guid,
                 claim_id,
                 num_sentence):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


        self.label_id = label_id
        self.guid = guid
        self.claim_id = claim_id
        self.num_sentence = num_sentence


# PROCESSOR
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
        
    @classmethod
    def _read_json(cls, input_file):
        lines = []
        with open(input_file, 'r') as f:
            reader = json.load(f)
            for idx in reader:
                line = list(reader[idx].values())
                lines.append(line)
        return lines

class FeverProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_pos_examples(self, data_dir):
        """See base class."""
        return self._create_examples_train(
            self._read_json(os.path.join(data_dir, "train_sup_sentences.json")), "train")

    def get_train_neg_examples(self, data_dir):
        """See base class."""
        return self._create_examples_train(
            self._read_json(os.path.join(data_dir, "train_refuted_sentences.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples_dev(
            self._read_json(os.path.join(data_dir, "dev_sentences.json")),
            "dev")

    def get_labels(self):
        """See base class."""
        return ["SUPPORTED", "REFUTED"]
    
    def get_info_eval(self, example):
        return example.guid, example.claim_id, example.num_sentence, example.claim, example.evidence

    def _create_examples_train(self, lines, set_type):
        """Creates examples for the training set."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            evidence = line[1]
            claim = line[2]
            label = line[3]
            domain = line[4]
            examples.append(
                InputExample_train(guid=guid, evidence=evidence, claim=claim, label=label,domain=domain))
        return examples
        
    def _create_examples_dev(self, lines, set_type):
        """Creates examples for the dev set and test set."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            claim_id = line[1] #evidence_address_page
            num_sentence = line[2] #evidence_address_sentence number
            evidence = line[3]
            claim = line[4]
            domain = line[5]
            label = line[6]
            examples.append(
                InputExample_dev(guid=guid, claim_id=claim_id, num_sentence=num_sentence, evidence=evidence, claim=claim, domain=domain, label=label))
        return examples


# INPUT FEATURE
def convert_examples_to_features_train(examples,
                                       label_list,
                                       max_seq_length,
                                       tokenizer,
                                       output_mode):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}
    # label_verification_map = {label: i for i, label in enumerate(label_verification_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Writing Example Train")):
        if ex_index % 100 == 0:
            # logger.info(f"\nHERE:\t{example.guid}")
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens_a = tokenizer.tokenize(example.evidence)

            tokens_b = None

            tokens_domain = tokenizer.tokenize(example.domain)

            # add domain 
            tokens_a = tokens_domain + tokens_a

        if example.claim:
            tokens_b = tokenizer.tokenize(example.claim)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example Train ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            print()

        features.append(
            InputFeatures_train(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          guid=example.guid))
    return features

def convert_examples_to_features_eval(examples,
                                      label_list,
                                      max_seq_length,
                                      tokenizer,
                                      output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    # label_verification_map = {label: i for i, label in enumerate(label_verification_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Writing Example Eval")):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.evidence)

        tokens_b = None

        tokens_domain = tokenizer.tokenize(example.domain)

        # add domain 
        tokens_a = tokens_domain + tokens_a
            
        if example.claim:
            tokens_b = tokenizer.tokenize(example.claim)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example Eval***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))
            print()
        features.append(
            InputFeatures_dev(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_id,
                                guid=example.guid,
                                claim_id=example.claim_id,
                                num_sentence=example.num_sentence))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    """
    Cách hoạt động:
        - Trong một vòng lặp vô hạn (while True), hàm tính tổng số token của cả hai
        chuỗi (total_length = len(tokens_a) + len(tokens_b)).
            + Nếu tổng số token không vượt quá max_length, hàm sẽ dừng lại và kết thúc.
            + Nếu tổng số token vượt quá max_length, hàm sẽ xác định chuỗi nào dài
            hơn và loại bỏ token cuối cùng của chuỗi đó (tokens_a.pop() hoặc tokens_b.pop()).
    """

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

# METRICS EVAL
def acc_and_f1(preds, labels):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average="macro")
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="vinai/phobert-base-v2", type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for positive samples.")
    parser.add_argument("--negative_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for negative samples.")
    parser.add_argument("--losstype",
                        default='cross_entropy',
                        type=str,
                        help="type of loss function.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()
    print('**********')
    print('**********')
    print(sys.argv[1:])
    print('**********')
    print('**********')

    processors = {
        "fever": FeverProcessor,
    }

    output_modes = {
        "fever": "classification",
    }

    # Huấn luyện song song trên GPU
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    # Xuất thông báo khởi tạo môi trường cho việc huấn luyện
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # Tích lũy gradient là một kỹ thuật hữu ích khi huấn luyện mô hình trên các
    # batch nhỏ. Thay vì cập nhật trọng số sau mỗi batch, gradient được tích lũy
    # qua nhiều bước, và chỉ cập nhật trọng số sau một số lượng nhất định của các
    # bước tích lũy. Điều này giúp làm giảm nhiễu trong quá trình cập nhật trọng
    # số và thường dẫn đến việc hội tụ nhanh hơn.
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)
    
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=False)
    # tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples_pos = processor.get_train_pos_examples(args.data_dir)
        train_examples_neg = processor.get_train_neg_examples(args.data_dir)
        train_examples_pos=train_examples_pos[0:200] #debugging
        train_examples_neg=train_examples_neg[0:400] #debugging
        num_train_optimization_steps = int(
            len(train_examples_pos) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        
        # logger.info(f"\nNum Train Optimization Steps: {num_train_optimization_steps}\n")

        if args.losstype == 'cross_entropy_concat' or args.losstype == 'hinge_loss_concat':
            num_train_optimization_steps = int(len(train_examples_pos+train_examples_neg) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    


    model = AutoModelForSequenceClassification.from_pretrained(args.bert_model,
                                                          cache_dir=cache_dir,
                                                          num_labels=num_labels,
                                                          from_tf=False)


    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=args.learning_rate)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    logger.info(f"Optimzer:{optimizer.__class__.__name__}")

    if args.do_train:
        train_features = convert_examples_to_features_train(
            train_examples_pos, label_list, args.max_seq_length,tokenizer, output_mode)

        train_features_neg = convert_examples_to_features_train(
            train_examples_neg, label_list, args.max_seq_length,tokenizer, output_mode)


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples_pos))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


        all_input_ids_neg = torch.tensor([f.input_ids for f in train_features_neg], dtype=torch.long)
        all_input_mask_neg = torch.tensor([f.input_mask for f in train_features_neg], dtype=torch.long)
        all_segment_ids_neg = torch.tensor([f.segment_ids for f in train_features_neg], dtype=torch.long)
        if output_mode == "classification":
            all_label_ids_neg = torch.tensor([f.label_id for f in train_features_neg], dtype=torch.long)

        train_data_neg = TensorDataset(all_input_ids_neg, all_input_mask_neg, all_segment_ids_neg, all_label_ids_neg)


        if args.losstype == 'pairwise_dependent':
            all_guids=torch.tensor([int(f.guid[6:((f.guid).find('_'))]) for f in train_features], dtype=torch.long)
            all_guids_neg =torch.tensor([int(f.guid[6:((f.guid).find('_'))]) for f in train_features_neg], dtype=torch.long)

            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_guids)


        if args.losstype == 'cross_entropy_concat' or args.losstype == 'hinge_loss_concat': #concate positive and negative pairs
            train_data = ConcatDataset([train_data, train_data_neg])

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
            train_neg_sampler = RandomSampler(train_data_neg)

        else:
            train_sampler = DistributedSampler(train_data)

        if args.losstype == 'pairwise_independent' or args.losstype =='cross_entropy_pairwise_independent': # no need for dependent
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,drop_last=True)
        else:
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        negative_dataloader = DataLoader(train_data_neg, sampler=train_neg_sampler, batch_size=args.negative_batch_size,drop_last=True)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            label_pred = []
            label_true = []
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # it=iter(negative_dataloader)
            it=iter(train_dataloader)
            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            for step, batch in enumerate(tqdm(negative_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                # input_ids, input_mask, segment_ids, label_ids = batch
                input_ids_neg, input_mask_neg, segment_ids_neg, label_ids_neg = batch

                # define a new function to compute loss values for both output_modes
                try:
                    # batch_neg = tuple(t.to(device) for t in next(it))
                    batch_train = tuple(t.to(device) for t in next(it))
                except:
                    # it = iter(negative_dataloader)
                    # batch_neg = tuple(t.to(device) for t in next(it))
                    it = iter(train_dataloader)
                    batch_train = tuple(t.to(device) for t in next(it))
                # input_ids_neg, input_mask_neg, segment_ids_neg, label_ids_neg = batch_neg
                input_ids, input_mask, segment_ids, label_ids = batch_train

                input_ids_cat=torch.cat([input_ids_neg, input_ids],dim=0)
                segment_ids_cat=torch.cat([segment_ids_neg, segment_ids],dim=0)
                input_mask_cat=torch.cat([input_mask_neg, input_mask],dim=0)
                label_ids_cat=torch.cat([label_ids_neg.view(-1), label_ids.view(-1)], dim = 0)

                with torch.no_grad():
                    output_logits = model(input_ids=input_ids_cat, token_type_ids=None, attention_mask=input_mask_cat, labels=label_ids_cat)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    if args.losstype == 'cross_entropy':
                        loss = loss_fct(output_logits.logits.view(-1, num_labels),label_ids_cat)

                    elif args.losstype == 'cross_entropy_mining':

                        loss_fct = CrossEntropyLoss(reduction='none')
                        loss = loss_fct(output_logits.logits.view(-1, num_labels),label_ids_cat)
                        REF_index = label_ids_cat.view(-1) == 1
                        SUP_index = label_ids_cat.view(-1) != 1
                        REF_numbers = REF_index.sum()
                        REF_numbers = 1

                        TOP_K, Hard_SUP_index = loss[SUP_index].topk(REF_numbers)
                        # loss = torch.cat([loss[SUP_index], TOP_K])    #uncomment if dont want to use no-grad then grad

                        #comment if dont want to use no-grad then grad
                        IDS = torch.cat([torch.tensor(range(0,REF_numbers),device=device), REF_numbers+Hard_SUP_index], dim=0).to(torch.int32)
                        output_logits = model(input_ids=input_ids_cat[IDS, :], attention_mask=input_mask_cat[IDS, :], labels=label_ids_cat[IDS])

                        # Compute loss, evaluation
                        probs = torch.nn.functional.softmax(output_logits.logits, dim=-1)
                        _, pred = torch.max(probs, dim=-1)

                        label_pred.extend(pred.cpu().numpy())
                        label_true.extend(label_ids_cat[IDS].cpu().numpy())

                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(output_logits.logits.view(-1, num_labels), label_ids_cat[IDS])

                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(output_logits.logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            print('printing loss')
            print('training_loss~=',tr_loss/nb_tr_steps)
            print(acc_and_f1(preds=label_pred, labels=label_true))

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_examples=eval_examples[0:100] #debugging
        eval_features = convert_examples_to_features_eval(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        # preds = []
        preds = []
        labels = []
        probs = []
        store_output=list()
        # softmaxing=torch.nn.Softmax()
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                # output_logits = model(input_ids, segment_ids, input_mask, labels=None)
                output_logits = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask, labels=label_ids)
                # logits=softmaxing(logits)
                store_output.extend(output_logits.logits.cpu().numpy())
                prob_in_batch = torch.nn.functional.softmax(output_logits.logits, dim=-1)

                # Chọn nhãn với xác suất cao nhất
                prob, pred = torch.max(prob_in_batch, dim=-1)
                preds.extend(pred.cpu().numpy())
                labels.extend(label_ids.cpu().numpy())
                probs.extend(prob.cpu().numpy())
        print(acc_and_f1(preds=preds, labels=labels))
        # torch.from_numpy(np.array(store_output))
        print("Storing dev scores")
        claim_ids = []
        num_sentences = []
        claims =[]
        evidences = []
        ids = []
        for example in eval_examples:
            id, claim_id, num_sentence, claim, evidence = processor.get_info_eval(example)
            ids.append(id)
            claim_ids.append(claim_id)
            num_sentences.append(num_sentence)
            claims.append(claim)
            evidences.append(evidence)
        
        data = {
            "claim_id": claim_ids,
            "num_sentence": num_sentences,
            "claim": claims,
            "evidence": evidences,
            "pred": preds,
            "label": labels,
            "prob": probs
        }
        eval_df = pd.DataFrame(data=data)
        eval_df.to_json(os.path.join(args.output_dir,"eval.json"), force_ascii=False, indent=4, orient='index')


    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)


if __name__ == "__main__":
    main()