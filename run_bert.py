from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('..')
import logging
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AlbertTokenizer, AlbertForMultipleChoice
from pytorch_pretrained_bert.optimization import BertAdam
import json
from utils_glue import convert_examples_to_features, compute_metrics, processors, \
    GLUE_TASKS_NUM_LABELS, MAX_SEQ_LENGTHS, output_modes

reverse_order = False
sa_step = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

'''
-------------------------------------------------------
Classes:
-------------------------------------------------------
InfiniteDataLoader: An iterator of dataset. Stop iteration if dataset ends, then reinitialize data loader
-------------------------------------------------------
Functions:
-------------------------------------------------------
1. set_seed: set random seed
2. train: Train the model, return step, training loss per step
3. evaluate: Evaluate the model,  return performance metrics
4. convert_features_to_tensors: convert features and return data tensor
5. load_and_cache_examples: Load data features from cache or dataset file, return dataset as tensors
-------------------------------------------------------
Usage of parameters:
--------------------------------------------------------
    data_dir: The input data dir for all tasks, separated by comma,
    bert_model: Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, 
                bert-base-cased, bert-base-multilingual, bert-base-chinese
    task_name: The name of the task to train in the list: mctest, race, dream
    output_dir: The output directory where the model checkpoints will be written
    do_lower_case: Whether to lower case the input text. True for uncased models, False for cased models
    max_seq_length: The maximum total input sequence length after WordPiece tokenization.
                    Sequences longer than this will be truncated, and sequences shorter than this will be padded
    do_train: Whether to run training
    do_eval: Whether to run eval on the dev set
    per_gpu_train_batch_size: Batch size per GPU/CPU for training
    per_gpu_eval_batch_size: Batch size per GPU/CPU for evaluation
    learning_rate: The initial learning rate for Adam
    max_grad_norm: Max gradient norm
    weight_decay: l2 regularization
    num_train_epochs: Total number of training epochs to perform
    warmup_proportion: Proportion of training to perform linear learning rate warmup for
    do_epoch_checkpoint: Save checkpoint at every epoch
    no_cuda: Whether not to use CUDA when available
    local_rank: local_rank for distributed training on gpus
    seed: random seed for initialization
    gradient_accumulation_steps: Number of updates steps to accumualte before performing a backward/update pass
'''
parameters = {
    'data_dir': None,
    'bert_model': "albert-large-v1",
    'task_name': "mctest,race,dream",
    'output_dir': "tmp/",

    'do_lower_case': True,
    'do_train': True,
    'do_eval': True,
    'per_gpu_train_batch_size': [8,8,8],
    'per_gpu_eval_batch_size': 16,
    'train_batch_size': None,
    'eval_batch_size': None,
    'learning_rate': 1e-5,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    'num_train_epochs': 3,
    'warmup_proportion': 0.1,
    'do_epoch_checkpoint': False,
    'no_cuda': False,
    'device': None,
    'n_gpu': None,
    'local_rank': -1,
    'seed': 42,
    'gradient_accumulation_steps': 1
}

parameters['data_dir'] = ["data/%s" % (dataset) for dataset in parameters['task_name'].upper().split(',')]

CONFIG_NAME = "config_{}.json".format(parameters['bert_model'])
WEIGHTS_NAME = "pytorch_model_{}.bin".format(parameters['bert_model'])

class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def get_next(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            self.data_iter = iter(self.data_loader)
            data = next(self.data_iter)
        return data

def set_seed(parameters):
    random.seed(parameters['seed'])
    np.random.seed(parameters['seed'])
    torch.manual_seed(parameters['seed'])
    if parameters['n_gpu'] > 0:
        torch.cuda.manual_seed_all(parameters['seed'])

def train(parameters, train_datasets, model, tokenizer):
    """ Train the model """
    parameters['train_batch_size'] = [per_gpu_train_batch_size * max(1, parameters['n_gpu'])
                             for per_gpu_train_batch_size in parameters['per_gpu_train_batch_size']]
    train_iters = []
    tr_batches = []
    for idx, train_dataset in enumerate(train_datasets):
        train_sampler = RandomSampler(train_dataset) if parameters['local_rank'] == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=parameters['train_batch_size'][idx])
        train_iters.append(InfiniteDataLoader(train_dataloader))
        tr_batches.append(len(train_dataloader))

    ## set sampling proportion
    total_n_tr_batches = sum(tr_batches)
    sampling_prob = [float(n_batches) / total_n_tr_batches for n_batches in tr_batches]

    t_total = total_n_tr_batches // parameters['gradient_accumulation_steps'] * parameters['num_train_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': parameters['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=parameters['learning_rate'],
                         warmup=parameters['warmup_proportion'],
                         max_grad_norm=parameters['max_grad_norm'],
                         t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", parameters['num_train_epochs'])
    logger.info("  Instantaneous batch size per GPU = %s", ','.join(map(str, parameters['per_gpu_train_batch_size'])))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   parameters['train_batch_size'][0] * parameters['gradient_accumulation_steps'] * (torch.distributed.get_world_size() if parameters['local_rank'] != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", parameters['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    nb_tr_examples = 0
    model.zero_grad()
    train_iterator = trange(int(parameters['num_train_epochs']), desc="Epoch", disable=parameters['local_rank'] not in [-1, 0])
    set_seed(parameters)  # Added here for reproductibility (even between python 2 and 3)
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(trange(total_n_tr_batches), desc="Iteration", disable=parameters['local_rank'] not in [-1, 0])
        for step, _ in enumerate(epoch_iterator):
            epoch_iterator.set_description("train loss: {}".format(tr_loss / nb_tr_examples if nb_tr_examples else tr_loss))
            model.train()

            # select task id
            task_id = np.argmax(np.random.multinomial(1, sampling_prob))
            batch = train_iters[task_id].get_next()

            batch = tuple(t.to(parameters['device']) for t in batch)

            inputs = {'input_ids':   batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':     torch.reshape(batch[3],(parameters['train_batch_size'][0],))}
            outputs = model(**inputs)
            loss = outputs[0]

            if parameters['n_gpu'] > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if parameters['gradient_accumulation_steps'] > 1:
                loss = loss / parameters['gradient_accumulation_steps']

            loss.backward()

            tr_loss += loss.item()

            nb_tr_examples += inputs['input_ids'].size(0)
            if (step + 1) % parameters['gradient_accumulation_steps'] == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

        if parameters['do_epoch_checkpoint']:
            epoch_output_dir = os.path.join(parameters['output_dir'], 'epoch_{}'.format(epoch))
            os.makedirs(epoch_output_dir, exist_ok=True)
            output_model_file = os.path.join(epoch_output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(epoch_output_dir, CONFIG_NAME)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(epoch_output_dir)

        evaluate(parameters, model, tokenizer, epoch=epoch, is_test=False)
        evaluate(parameters, model, tokenizer, epoch=epoch, is_test=True)

    return global_step, tr_loss / global_step

def evaluate(parameters, model, tokenizer, epoch=0, is_test=False):

    eval_task_names = parameters['task_name']
    eval_output_dir = parameters['output_dir']

    set_type = 'test' if is_test else 'dev'
    results = {}
    for task_id, eval_task in enumerate(eval_task_names):
        if is_test and not hasattr(processors[eval_task], 'get_test_examples'):
            continue

        eval_dataset = load_and_cache_examples(parameters, eval_task, tokenizer, set_type)

        if not os.path.exists(eval_output_dir) and parameters['local_rank'] in [-1, 0]:
            os.makedirs(eval_output_dir)

        parameters['eval_batch_size'] = parameters['per_gpu_eval_batch_size'] * max(1, parameters['n_gpu'])
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if parameters['local_rank'] == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=parameters['eval_batch_size'])

        # Eval!
        logger.info("***** Running evaluation for {} on {} for epoch {} *****".format(eval_task, set_type, epoch))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", parameters['eval_batch_size'])
        eval_loss = 0.0
        nb_eval_steps = 0
        logits_all = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            
            batch = tuple(t.to(parameters['device']) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],  # XLM don't use segment_ids
                      'labels':     torch.reshape(batch[3],(parameters['eval_batch_size'],))
                          }
              
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if logits_all is None:
                logits_all = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                logits_all = np.append(logits_all, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        output_mode = output_modes[eval_task]
        if output_mode in ["classification", "multi-choice"]:
            preds = np.argmax(logits_all, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(logits_all)
        result = compute_metrics(eval_task, preds, out_label_ids.reshape(-1))
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results_{}_{}.txt".format(eval_task, set_type))
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results for {} on {} for epoch {} *****".format(eval_task, set_type, epoch))
            writer.write("***** Eval results for epoch {} *****\n".format(epoch))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            logger.info("\n")

        # get error idx
        correct_idx = np.argwhere(preds == out_label_ids).tolist()
        wrong_idx = np.argwhere(preds != out_label_ids).tolist()
        wrong_idx_dict = {'correct': correct_idx, 'wrong': wrong_idx,
                  'preds': preds.tolist(), 'logits': logits_all.tolist(),
                  'labels': out_label_ids.tolist()}
        json.dump(wrong_idx_dict, open(os.path.join(eval_output_dir,
                                                    "error_idx_{}_{}.json".format(eval_task, set_type)), 'w'))

    return results

def convert_features_to_tensors(features, output_mode, is_multi_choice=True):

    input_ids = []
    input_mask = []
    segment_ids = []
    label_id = []

    if is_multi_choice:
        n_class = len(features[0])
        for f in features:
            input_ids.append([])
            input_mask.append([])
            segment_ids.append([])
            for i in range(n_class):
                input_ids[-1].append(f[i].input_ids)
                input_mask[-1].append(f[i].input_mask)
                segment_ids[-1].append(f[i].segment_ids)

            label_id.append([f[0].label_id])
    else:
        for f in features:
            input_ids.append(f.input_ids)
            input_mask.append(f.input_mask)
            segment_ids.append(f.segment_ids)
            label_id.append(f.label_id)

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)

    if output_mode in ["classification", "multi-choice"]:
        all_label_ids = torch.tensor(label_id, dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(label_id, dtype=torch.float)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return data

def load_and_cache_examples(parameters, task, tokenizer, set_type='train'):
    processor = processors[task]()
    output_mode = output_modes[task]
    is_multi_choice = True if output_mode == 'multi-choice' else False
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(parameters['data_dir'][task], 'cached_{}_{}_{}'.format(
        set_type,
        list(filter(None, parameters['bert_model'].split('/'))).pop(),
        str(MAX_SEQ_LENGTHS[task]),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", parameters['data_dir'][task])
        label_list = processor.get_labels()
        if set_type == 'train':
            examples = processor.get_train_examples(parameters['data_dir'][task])
        elif set_type == 'dev':
            examples = processor.get_dev_examples(parameters['data_dir'][task])
        else:
            examples = processor.get_test_examples(parameters['data_dir'][task])
        features = convert_examples_to_features(examples, label_list, MAX_SEQ_LENGTHS[task],
                                                tokenizer,
                                                output_mode=output_mode,
                                                do_lower_case=parameters['do_lower_case'],
                                                is_multi_choice=is_multi_choice)
        if parameters['local_rank'] in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    dataset = convert_features_to_tensors(features, output_mode, is_multi_choice=is_multi_choice)

    return dataset

def main():
    if parameters['local_rank'] == -1 or parameters['no_cuda']:
        parameters['device'] = torch.device("cuda:0" if torch.cuda.is_available() and not parameters['no_cuda'] else "cpu")
        parameters['n_gpu'] = torch.cuda.device_count()
    else:
        parameters['device'] = torch.device("cuda:0", parameters['local_rank'])
        parameters['n_gpu'] = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", parameters['device'], parameters['n_gpu'], bool(parameters['local_rank'] != -1))

    if parameters['gradient_accumulation_steps'] < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            parameters['gradient_accumulation_steps']))

    if not parameters['do_train'] and not parameters['do_eval']:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(parameters['output_dir']) and os.listdir(parameters['output_dir']):
        if parameters['do_train']:
            print("Output directory ({}) already exists and is not empty.".format(parameters['output_dir']))
    else:
        os.makedirs(parameters['output_dir'], exist_ok=True)

    set_seed(parameters)

    ## prepare tasks
    # split and lowercase task names
    parameters['task_name'] = parameters['task_name'].lower().split(',')
    for task_name in parameters['task_name']:
        if task_name not in processors:
            raise ValueError("Task not found: %s" % (parameters['task_name']))
    parameters['data_dir'] = {task_name: data_dir_ for task_name, data_dir_ in zip(parameters['task_name'], parameters['data_dir'])}
    num_labels = [GLUE_TASKS_NUM_LABELS[task_name] for task_name in parameters['task_name']]
    task_output_config = [(output_modes[task_name], num_label)
                          for task_name, num_label in zip(parameters['task_name'], num_labels)]

    tokenizer = AlbertTokenizer.from_pretrained(parameters['bert_model'])
    model = AlbertForMultipleChoice.from_pretrained(parameters['bert_model'])
    model.to(parameters['device'])

    if parameters['local_rank'] != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=parameters['local_rank'],
                                                          output_device=parameters['local_rank'])
    elif parameters['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    if parameters['do_train']:
        train_datasets = [load_and_cache_examples(parameters, task_name, tokenizer, set_type='train')
                          for task_name in parameters['task_name']]
        global_step, tr_loss = train(parameters, train_datasets, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # final save of model parameters
        output_model_file = os.path.join(parameters['output_dir'], WEIGHTS_NAME)
        output_config_file = os.path.join(parameters['output_dir'], CONFIG_NAME)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(parameters['output_dir'])

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    if parameters['do_eval'] and not parameters['do_train']:
        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(os.path.join(parameters['output_dir'], WEIGHTS_NAME),map_location=map_location),strict=False)
        else:
            model.load_state_dict(torch.load(os.path.join(parameters['output_dir'], WEIGHTS_NAME),map_location=map_location),strict=False)

        model.eval()
        epoch = parameters['num_train_epochs']
        evaluate(parameters, model, tokenizer, epoch=epoch, is_test=False)
        evaluate(parameters, model, tokenizer, epoch=epoch, is_test=True)

if __name__ == "__main__":
    main()
