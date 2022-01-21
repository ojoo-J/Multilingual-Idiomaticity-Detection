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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import torch
import re

from scipy.stats import pearsonr, spearmanr, truncnorm
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
)
import random
import nltk
from nltk.corpus import wordnet

logger = logging.getLogger(__name__)



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
        self,
        guid,
        text_a,
        text_prev,
        text_next,
        mwe,
        text_b=None,
        label=None,
        POS=None,
        FGPOS=None,
        text_a_2=None,
        text_b_2=None,
    ):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_prev = text_prev
        self.text_next = text_next
        self.mwe = mwe
        self.text_b = text_b
        self.label = label
        self.POS = POS
        self.FGPOS = FGPOS
        self.text_a_2 = text_a_2
        self.text_b_2 = text_b_2

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_id,
        guid=None,
        input_ids_2=None,
        input_mask_2=None,
        segment_ids_2=None,
        input_ids_3=None,
        input_mask_3=None,
        segment_ids_3=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
        self.input_ids_2 = input_ids_2
        self.input_mask_2 = input_mask_2
        self.segment_ids_2 = segment_ids_2
        self.input_ids_3 = input_ids_3
        self.input_mask_3 = input_mask_3
        self.segment_ids_3 = segment_ids_3



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_eval_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='UTF-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines


class TrofiProcessor(DataProcessor):
    """Processor for the TroFi and MOH-X data set."""

    def get_train_examples(self, data_dir, k=None):
        """See base class."""
        if k is not None:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "train" + str(k) + ".csv")), "train"
            )
        else:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "train.csv")), "train"
            )

    def get_test_examples(self, data_dir, k=None):
        """See base class."""
        if k is not None:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "test" + str(k) + ".csv")), "test"
            )
        else:
            return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_dev_examples(self, data_dir, k=None):
        """See base class."""
        if k is not None:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "dev" + str(k) + ".csv")), "dev"
            )
        else:
            return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[2]
            label = line[1]
            POS = line[3]
            FGPOS = line[4]
            index = line[-1]
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_b=index, label=label, POS=POS, FGPOS=FGPOS
                )
            )
        return examples

class IdiomProcessor(DataProcessor):
    """Processor for the Idiom data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_eval_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "eval.csv")), "eval")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, line[0])
            text_a = line[2]
            text_prev = line[3]
            text_next = line[4]
            mwe = line[-2]
            label = line[1]
            # POS = line[3]
            # FGPOS = line[4]
            index = line[-1]
            examples.append(
                InputExample(
                    guid=guid, text_a=text_a, text_prev=text_prev, text_next=text_next, mwe=mwe, text_b=index, label=label
                )
            )
        return examples

### MELBERT_CONTEXT
def convert_examples_to_two_features_with_context(
    examples, label_list, max_seq_length, tokenizer, output_mode, args
):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)  # tokenize the sentence
        tokens_prev = tokenizer.tokenize(example.text_prev)
        tokens_next = tokenizer.tokenize(example.text_next)
        tokens_mwe = tokenizer.tokenize(example.mwe)
        tokens_all = tokens_prev + tokens_a + tokens_next
        tokens_b = None
        mwe_index = None
        tokens_no_mwe_target = []

        text_b_list=[]

        index_yes = True

        try: 

            text_b_str = example.text_b.split(" ")
            text_b = [int(x) for x in text_b_str]
        
            #text_b = list(map(int, text_b))  # index of target word = list

            #print('text_b: ', text_b)


            tokens_b = int(text_b[0])
            mwe_index = int(text_b[0])
        
                # truncate the sentence to max_seq_len
                #if len(tokens_all) > max_seq_length - 6:
                #    tokens_a = tokens_a[: (max_seq_length - 6)]

                

            # Find the target word index
            for i, w in enumerate(example.text_a.split()):
                # If w is a target word, tokenize the word and save to text_b
                #print('i의 type:', type(i))
                if i in list(text_b): # index에 해당하는 word만 토큰나이징
                
                    # consider the index due to models that use a byte-level BPE as a tokenizer (e.g., GPT2, RoBERTa)
                    #text_b = tokenizer.tokenize(w) if i == 0 else tokenizer.tokenize(" " + w)
                    #break
                    if i==0:
                        text_b_list.append(tokenizer.tokenize(w))
                    else:
                        text_b_list.append(tokenizer.tokenize(" " + w))
    
                #print('text_b_list: ', text_b_list)

                # text_a의 모든 word에 대해 토크나이징
                w_tok = tokenizer.tokenize(w) if i == 0 else tokenizer.tokenize(" " + w)
                #print(ex_index,'번째 example의', i, '번째 word의 w_tok:', w_tok)

                # Count number of tokens before the target word to get the target word index
                #if w_tok:
                    #tokens_b += len(w_tok) - 1
                #print('mwe_index의 type:', type(mwe_index))
                if i < mwe_index:
                    tokens_b += len(w_tok)-1 # mwe_index + mwe_index 전까지의 토큰 수 = 토크나이징 후의 mwe index

            ## prev 추가
            for i, w in enumerate(example.text_prev.split()):
                prev_tok = tokenizer.tokenize(w) if i == 0 else tokenizer.tokenize(" " + w)
                tokens_b += len(prev_tok) # prev 토큰 개수만큼 mwe index 뒤로 미루기
                

            #except TypeError:
            #    if example.text_b:
            #        tokens_b = tokenizer.tokenize(example.text_b)
            #        # Account for [CLS], [SEP], [SEP] with "- 3"
            #        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            #    else:
            #        # Account for [CLS] and [SEP] with "- 2"
            #        if len(tokens_a) > max_seq_length - 2:
            #            tokens_a = tokens_a[: (max_seq_length - 2)]
        except: 
            index_yes = False


        if (len(tokens_all)+len(sum(text_b_list, []))) > max_seq_length - 6:
                tokens_all = tokens_all[: (max_seq_length - 6 - len(sum(text_b_list, [])))]
        

        tokens = [tokenizer.cls_token] + tokens_all + sum(text_b_list, []) + [tokenizer.sep_token]
        #print('tokens:', tokens)
        # POS tag tokens
        if False:
            POS_token = tokenizer.tokenize(example.POS)
            tokens += POS_token + [tokenizer.sep_token]

        # Local context
        if False:
            local_start = 1
            local_end = local_start + len(tokens_a)
            comma1 = tokenizer.tokenize(",")[0]
            comma2 = tokenizer.tokenize(" ,")[0]
            for i, w in enumerate(tokens):
                if i < tokens_b + 1 and (w in [comma1, comma2]):
                    local_start = i
                if i > tokens_b + 1 and (w in [comma1, comma2]):
                    local_end = i
                    break
            segment_ids = [
                2 if i >= local_start and i <= local_end else 0 for i in range(len(tokens))
            ]
        else:
            segment_ids = [0] * len(tokens)

        # POS tag encoding
        after_token_a = False
        for i, t in enumerate(tokens):
            if t == tokenizer.sep_token:
                after_token_a = True
            if after_token_a and t != tokenizer.sep_token:
                segment_ids[i] = 3

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if index_yes==True:
            tokens_b += 1  # add 1 to the target word index considering [CLS]
            try:
                for i in range(len(sum(text_b_list, []))): # text_b -> text_b_list
                    segment_ids[tokens_b + i] = 1
            except:
                pass

        #print('segment_ids check: ', segment_ids)


        input_mask = [1] * len(input_ids)
        padding = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
            max_seq_length - len(input_ids)
        )
        input_ids += padding
        input_mask += [0] * len(padding)
        segment_ids += [0] * len(padding)
        
        #print('input_ids: ',input_ids)
        #print('input_mask: ',input_mask)
        #print('segment_ids: ',segment_ids)
        #print('len(input_ids): ', len(input_ids))
        #print('len(input_mask): ', len(input_mask))
        #print('len(segment_ids): ', len(segment_ids))

        #print('len(input_ids): ', len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[str(example.label)]
        else:
            raise KeyError(output_mode)


        ### Third features

        for token in tokens_a:
            if token not in sum(text_b_list, []):
                tokens_no_mwe_target.append(token)
            else:
                tokens_no_mwe_target.append(tokenizer.mask_token)


        #print('mwe: ', sum(text_b_list, []))
        #print('target 원본: ', tokens_a)
        #print('mwe 제거한 target: ', tokens_no_mwe_target)

        tokens_all = tokens_no_mwe_target

        if len(tokens_all) > max_seq_length - 6:
            tokens_all = tokens_all[: (max_seq_length - 6)]

        tokens = [tokenizer.cls_token] + tokens_all + [tokenizer.sep_token]

        segment_ids_3 = [0] * len(tokens)
        #tokens_b = 1  # add 1 to the target word index considering [CLS]
        #for i in range(len(tokens_all)):
        #    segment_ids_3[tokens_b + i] = 1


        input_ids_3 = tokenizer.convert_tokens_to_ids(tokens)
        input_mask_3 = [1] * len(input_ids_3)
        padding = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
            max_seq_length - len(input_ids_3)
        )
        input_ids_3 += padding
        input_mask_3 += [0] * len(padding)
        segment_ids_3 += [0] * len(padding)

        assert len(input_ids_3) == max_seq_length
        assert len(input_mask_3) == max_seq_length
        assert len(segment_ids_3) == max_seq_length



        # Second features (Target word)
        tokens = [tokenizer.cls_token] + tokens_mwe + [tokenizer.sep_token]
        #print('tokenizer.cls_token: ', [tokenizer.cls_token])
        #print('[tokenizer.sep_token]: ', [tokenizer.sep_token])
        #print('tokens: ', tokens)
        #print('text_b_list: ', sum(text_b_list,[]))

        segment_ids_2 = [0] * len(tokens)
        tokens_b = 1  # add 1 to the target word index considering [CLS]
        for i in range(len(tokens_mwe)):
            segment_ids_2[tokens_b + i] = 1

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_ids_2 = tokenizer.convert_tokens_to_ids(tokens)
        input_mask_2 = [1] * len(input_ids_2)

        padding = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
            max_seq_length - len(input_ids_2)
        )
        input_ids_2 += padding
        input_mask_2 += [0] * len(padding)
        segment_ids_2 += [0] * len(padding)

        
        assert len(input_ids_2) == max_seq_length
        assert len(input_mask_2) == max_seq_length
        assert len(segment_ids_2) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                guid=example.guid + " " + str(example.text_b),
                input_ids_2=input_ids_2,
                input_mask_2=input_mask_2,
                segment_ids_2=segment_ids_2,
                input_ids_3=input_ids_3,
                input_mask_3=input_mask_3,
                segment_ids_3=segment_ids_3,
            )
        )

    return features



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def seq_accuracy(preds, labels):
    acc = []
    for idx, pred in enumerate(preds):
        acc.append((pred == labels[idx]).mean())
    return acc.mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1_macro": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def all_metrics(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    pre = precision_score(y_true=labels, y_pred=preds)
    rec = recall_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": pre,
        "recall": rec,
        "f1_macro": f1,
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return all_metrics(preds, labels)


processors = {
    "idiom": IdiomProcessor,
    "trofi": TrofiProcessor,
}

output_modes = {
    "idiom": "classification",
    "trofi": "classification",
}
