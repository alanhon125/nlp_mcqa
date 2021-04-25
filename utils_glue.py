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
""" BERT classification fine-tuning (modified for just MCTest, RACE and DREAM): utilities to work with General Language Understanding Evaluation (GLUE) benchmark tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import sys
from io import open
import json
from os import listdir
from os.path import isfile, join

from random import randint
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

output_modes = {
    "dream": "multi-choice",
    "race": 'multi-choice',
    "mctest": 'multi-choice',
    "mctest160": 'multi-choice',
    "mctest500": 'multi-choice',
}

GLUE_TASKS_NUM_LABELS = {
    "dream": 3,
    "race": 4,
    "mctest": 4,
    "mctest160": 4,
    "mctest500": 4,
}

MAX_SEQ_LENGTHS = {
    "race": 256,
    "dream": 256,
    "mctest": 256,
    "mctest160": 256,
    "mctest500": 256,
}

'''
-------------------------------------------------------
Classes:
-------------------------------------------------------
1. InputExample: A single training/test example for simple sequence classification
2. InputFeatures: A single set of features of data
3. DataProcessor: Base class for data converters for sequence classification data sets
4. DreamProcessor
5. RaceProcessor
6. MCTest160Processor
7. MCTest500Processor
8. MCTestProcessor

Functions summary:
1. convert_examples_to_features: Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
2. _truncate_seq_tuple: Truncates a sequence tuple in place to the maximum length
3. _truncate_seq_pair: Truncates a sequence pair in place to the maximum length
4. simple_accuracy: return accuracy
5. acc_and_f1: return dictionary of acc and f1
6. pearson_and_spearman: return dictionary of Pearson and Spearman correlation coefficients
7. compute_metrics: compute metrics (you can customize to show accuracy/acc_and_f1/pearson_and_spearman)
'''


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
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
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


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
    def _read_tsv(cls, input_file, quotechar=None, remove_header=False):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            if remove_header:
                next(reader)
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class DreamProcessor(DataProcessor):

    def get_train_examples(self, data_dir, level=None):
        """See base class."""
        return self._create_examples(
            data_dir, "train", level=level)

    def get_dev_examples(self, data_dir, level=None):
        """See base class."""
        return self._create_examples(
            data_dir, "dev", level=level)

    def get_test_examples(self, data_dir, level=None):
        """See base class."""
        return self._create_examples(
            data_dir, "test", level=level)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, data_dir, set_type, level=None):
        """Creates examples for the training and dev sets."""
        examples = []
        with open(data_dir + "/{}.json".format(set_type), 'r') as f:
            data = json.load(f)
            if level is None:
                for i in range(len(data)):
                    for j in range(len(data[i][1])):
                        text_a = '\n'.join(data[i][0])
                        text_c = data[i][1][j]["question"]
                        options = []
                        for k in range(len(data[i][1][j]["choice"])):
                            options.append(data[i][1][j]["choice"][k])
                        answer = data[i][1][j]["answer"]
                        label = str(options.index(answer))
                        for k in range(len(options)):
                            guid = "%s-%s-%s" % (set_type, i, k)
                            examples.append(
                                InputExample(guid=guid, text_a=text_a, text_b=options[k], label=label, text_c=text_c))
            else:
                i = randint(0, len(data) - 1)
                logger.info("*** Drawing example ***")
                logger.info(f"example_id: {i}")
                logger.info(f"Passage: {data[i][0]}")
                for j in range(len(data[i][1])):
                    text_a = '\n'.join(data[i][0])
                    text_c = data[i][1][j]["question"]
                    logger.info(f"Question: {text_c}")
                    options = []
                    for k in range(len(data[i][1][j]["choice"])):
                        options.append(data[i][1][j]["choice"][k])
                    logger.info(f"Choice: {options}")
                    answer = data[i][1][j]["answer"]
                    label = str(options.index(answer))
                    for k in range(len(options)):
                        guid = "%s-%s-%s" % (set_type, i, k)
                        examples.append(
                            InputExample(guid=guid, text_a=text_a, text_b=options[k], label=label, text_c=text_c))
        return examples


class RaceProcessor(DataProcessor):

    def get_train_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "train", level=level)

    def get_test_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "test", level=level)

    def get_dev_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "dev", level=level)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_dataset_name(self):
        return 'RACE'

    def _read_samples(self, data_dir, set_type, level=None):
        # if self.level == None:
        #     data_dirs = ['{}/{}/{}'.format(data_dir, set_type, 'high'),
        #                  '{}/{}/{}'.format(data_dir, set_type, 'middle')]
        # else:
        # data_dirs = ['{}/{}/{}'.format(data_dir, set_type, self.level)]
        if level is None:
            data_dirs = ['{}/{}/{}'.format(data_dir, set_type, 'high'),
                         '{}/{}/{}'.format(data_dir, set_type, 'middle')]
        else:
            data_dirs = ['{}/{}/{}'.format(data_dir, set_type, level)]

        examples = []
        example_id = 0
        for data_dir in data_dirs:
            # filenames = glob.glob(data_dir + "/*txt")
            filenames = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
            for filename in filenames:
                with open(filename, 'r', encoding='utf-8') as fpr:
                    data_raw = json.load(fpr)
                    article = data_raw['article']
                    for i in range(len(data_raw['answers'])):
                        example_id += 1
                        truth = str(ord(data_raw['answers'][i]) - ord('A'))
                        question = data_raw['questions'][i]
                        options = data_raw['options'][i]
                        for k in range(len(options)):
                            guid = "%s-%s-%s" % (set_type, example_id, k)
                            option = options[k]
                            examples.append(
                                InputExample(guid=guid, text_a=article, text_b=option, label=truth,
                                             text_c=question))

        return examples


class MCTest160Processor(DataProcessor):

    def get_train_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "train", level=level)

    def get_test_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "test", level=level)

    def get_dev_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "dev", level=level)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_dataset_name(self):
        return 'MCTest160'

    def _read_samples(self, data_dir, set_type, level=None):

        with open(join(data_dir, "mc160.{}.tsv".format(set_type)), 'r', encoding='utf-8') as fpr:
            articles = []
            questions = [[]]
            options = [[]]
            for line in fpr:
                line = line.strip().split('\t')
                assert len(line) == 23
                articles.append(line[2].replace("\\newline", " "))
                for idx in range(3, 23, 5):
                    questions[-1].append(line[idx].partition(":")[-1][1:])
                    options[-1].append(line[idx + 1:idx + 5])
                questions.append([])
                options.append([])

        with open(join(data_dir, "mc160.{}.ans".format(set_type)), 'r', encoding='utf-8') as fpr:
            answers = []
            for line in fpr:
                line = line.strip().split('\t')
                answers.append(list(map(lambda x: str(ord(x) - ord('A')), line)))

        examples = []
        example_id = 0
        article_id = 0
        rand_id = randint(0, len(articles) - 1)

        if level is None:
            for article, question, option, answer in zip(articles, questions, options, answers):
                for ques, opt, ans in zip(question, option, answer):
                    example_id += 1
                    for k, op in enumerate(opt):
                        guid = "%s-%s-%s" % (set_type, example_id, k)
                        examples.append(
                            InputExample(guid=guid, text_a=article, text_b=op, label=ans,
                                         text_c=ques))
        else:
            for article, question, option, answer in zip(articles, questions, options, answers):
                article_id += 1
                if article_id != rand_id:
                    continue
                logger.info("*** Drawing example ***")
                logger.info(f"article_id: {article_id - 1}")
                for ques, opt, ans in zip(question, option, answer):
                    example_id += 1
                    for k, op in enumerate(opt):
                        guid = "%s-%s-%s" % (set_type, example_id, k)
                        examples.append(
                            InputExample(guid=guid, text_a=article, text_b=op, label=ans,
                                         text_c=ques))

        return examples


class MCTest500Processor(DataProcessor):

    def get_train_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "train", level=level)

    def get_test_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "test", level=level)

    def get_dev_examples(self, data_dir, level=None):
        """See base class."""
        return self._read_samples(data_dir, "dev", level=level)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_dataset_name(self):
        return 'MCTest500'

    def _read_samples(self, data_dir, set_type, level=None):

        with open(join(data_dir, "mc500.{}.tsv".format(set_type)), 'r', encoding='utf-8') as fpr:
            articles = []
            questions = [[]]
            options = [[]]
            for line in fpr:
                line = line.strip().split('\t')
                assert len(line) == 23
                articles.append(line[2].replace("\\newline", " "))
                for idx in range(3, 23, 5):
                    questions[-1].append(line[idx].partition(":")[-1][1:])
                    options[-1].append(line[idx + 1:idx + 5])
                questions.append([])
                options.append([])

        with open(join(data_dir, "mc500.{}.ans".format(set_type)), 'r', encoding='utf-8') as fpr:
            answers = []
            for line in fpr:
                line = line.strip().split('\t')
                answers.append(list(map(lambda x: str(ord(x) - ord('A')), line)))

        examples = []
        example_id = 0
        article_id = 0
        rand_id = randint(0, len(articles) - 1)

        if level is None:
            for article, question, option, answer in zip(articles, questions, options, answers):
                for ques, opt, ans in zip(question, option, answer):
                    example_id += 1
                    for k, op in enumerate(opt):
                        guid = "%s-%s-%s" % (set_type, example_id, k)
                        examples.append(
                            InputExample(guid=guid, text_a=article, text_b=op, label=ans,
                                         text_c=ques))
        else:
            for article, question, option, answer in zip(articles, questions, options, answers):
                article_id += 1
                if article_id != rand_id:
                    continue
                logger.info("*** Drawing example ***")
                logger.info(f"article_id: {article_id - 1}")
                for ques, opt, ans in zip(question, option, answer):
                    example_id += 1
                    for k, op in enumerate(opt):
                        guid = "%s-%s-%s" % (set_type, example_id, k)
                        examples.append(
                            InputExample(guid=guid, text_a=article, text_b=op, label=ans,
                                         text_c=ques))

        return examples


class MCTestProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._read_samples(data_dir, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def get_dataset_name(self):
        return 'MCTest'

    def _read_samples(self, data_dir, set_type):

        articles = []
        questions = [[]]
        options = [[]]
        for filename in [join(data_dir, "mc160.{}.tsv".format(set_type)),
                         join(data_dir, "mc500.{}.tsv".format(set_type))]:
            with open(filename, 'r', encoding='utf-8') as fpr:
                for line in fpr:
                    line = line.strip().split('\t')
                    assert len(line) == 23
                    articles.append(line[2].replace("\\newline", " "))
                    for idx in range(3, 23, 5):
                        questions[-1].append(line[idx].partition(":")[-1][1:])
                        options[-1].append(line[idx + 1:idx + 5])
                    questions.append([])
                    options.append([])

        answers = []
        for filename in [join(data_dir, "mc160.{}.ans".format(set_type)),
                         join(data_dir, "mc500.{}.ans".format(set_type))]:
            with open(filename, 'r', encoding='utf-8') as fpr:
                for line in fpr:
                    line = line.strip().split('\t')
                    answers.append(list(map(lambda x: str(ord(x) - ord('A')), line)))

        examples = []
        example_id = 0
        for article, question, option, answer in zip(articles, questions, options, answers):
            for ques, opt, ans in zip(question, option, answer):
                example_id += 1
                for k, op in enumerate(opt):
                    guid = "%s-%s-%s" % (set_type, example_id, k)
                    examples.append(
                        InputExample(guid=guid, text_a=article, text_b=op, label=ans,
                                     text_c=ques))

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 do_lower_case=False, is_multi_choice=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    if is_multi_choice:
        features = [[]]
    else:
        features = []
    for (ex_index, example) in enumerate(examples):
        if do_lower_case:
            example.text_a = example.text_a.lower()
            example.text_b = example.text_b.lower()
            example.text_c = example.text_c.lower()

        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None
        if example.text_b and example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        elif example.text_b and not example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
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
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
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
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_c:
            tokens_b += [sep_token] + tokens_c

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode in ["classification", "multi-choice"]:
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if is_multi_choice:
            features[-1].append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id))
            if len(features[-1]) == num_labels:
                features.append([])
        else:
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id))

    if is_multi_choice:
        if len(features[-1]) == 0:
            features = features[:-1]

    return features


def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


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


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
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


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == 'race':
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'dream':
        return {"acc": simple_accuracy(preds, labels)}
    else:
        return {"acc": simple_accuracy(preds, labels)}


processors = {
    "dream": DreamProcessor,
    "race": RaceProcessor,
    "mctest": MCTestProcessor,
    "mctest160": MCTest160Processor,
    "mctest500": MCTest500Processor,
}