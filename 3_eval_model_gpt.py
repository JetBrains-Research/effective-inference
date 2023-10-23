import os

import argparse
import yaml
import h5py
import numpy as np
import torch
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils.dataset_cache import cache_embeddings, get_dataset_for_regression, build_dataset_from_cached, load_cached_dataset
from utils.dataset_cache import build_dict_dataset_from_cached
from utils.prepare_dataset import load_datasets, cut_datasets
from utils.config import ConfigWrapper
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, List, Dict, Optional, Union
from numpy.random import shuffle
from sklearn.metrics import r2_score

from IPython.display import clear_output
from collections import defaultdict

from utils.attentions.bert.linear import BertWrapperLin, LinearClassifierBertAttention, LinearAttention
from utils.dataset_utils import get_dict_batch, prepare_batches
from utils.train_linear_utils import train_epoch, eval_epoch, plot_history

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report

def read_squad(path, n=None):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    # initialize lists for contexts, questions, and answers
    contexts = []
    questions = []
    answers = []
    # iterate through all data in squad data
    if n:
        t = max(n//len(squad_dict['data']), 1)
        for group in squad_dict['data']:
            for passage in group['paragraphs'][:t]:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    if 'plausible_answers' in qa.keys():
                        access = 'plausible_answers'
                    else:
                        access = 'answers'
                    for answer in qa['answers']:
                        # append data to lists
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)
        contexts = contexts[:n]
        questions = questions[:n]
        answers = answers[:n]
        
    else:
        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    if 'plausible_answers' in qa.keys():
                        access = 'plausible_answers'
                    else:
                        access = 'answers'
                    for answer in qa['answers']:
                        # append data to lists
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)
    # return formatted data lists
    return contexts, questions, answers
    
def add_end_idx(answers, contexts):
    # loop through each answer-context pair
    for answer, context in zip(answers, contexts):
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer['text']
        # we already know the start index
        start_idx = answer['answer_start']
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)

        # ...however, sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            # if the answer is not off :)
            answer['answer_end'] = end_idx
        else:
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    # this means the answer is off by 'n' tokens
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n  
                    
def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(encodings)):
        # append start/end token position using char_to_token method
        start_positions.append(answers[i]['answer_start'])
        end_positions.append(answers[i]['answer_end'])
        
        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift one token forward
        go_back = 1
        while end_positions[-1] is None:
            end_positions[-1] = answers[i]['answer_end']-go_back
            go_back +=1
        encodings[i].update({'start_positions': [start_positions[-1]], 'end_positions': [end_positions[-1]]})
    # update our encodings object with the new token-based start/end positions

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.encodings[idx].items()}

    def __len__(self):
        return len(self.encodings)

tqdm_pbar = lambda x, y: tqdm(x, leave=True, position=0, total=len(x), desc=f'{y}')
def get_squad_dataset(config, tokenizer, model,
                                   pbar_func=tqdm_pbar):
    n_train = config.data.cut_size
    train_contexts, train_questions, train_answers = read_squad('squad/train-v2.0.json', n_train)
    val_contexts, val_questions, val_answers = read_squad('squad/dev-v2.0.json', n_train)

    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    train_encodings = []
    idxs = list(range(len(train_contexts)))
    shuffle(idxs)
    
    for c, q in zip([train_contexts[i] for i in idxs[:n_train]], [train_questions[i] for i in idxs[:n_train]]):
      train_encodings.append(tokenizer(c, q, truncation=True, padding=True, max_length=256, return_tensors="pt"))
    
    val_encodings = []
    idxs = list(range(len(val_contexts)))
    shuffle(idxs)
    for c, q in zip([val_contexts[i] for i in idxs[:1000]], [val_questions[i] for i in idxs[:1000]]):
      val_encodings.append(tokenizer(c, q, truncation=True, padding=True, max_length=256, return_tensors="pt"))
    
    add_token_positions(train_encodings, train_answers)
    add_token_positions(val_encodings, val_answers)
    return SquadDataset(train_encodings), SquadDataset(val_encodings)

def finetune(train_dataset, model):
    optim = AdamW(model.parameters(), lr=5e-5)
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    for epoch in range(batch_size=config.data.num_epochs):
        # set model to train mode
        model.train()
        # setup loop (we use tqdm for the progress bar)
        loop = tqdm(train_dataset, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all the tensor batches required for training
            input_ids = batch['input_ids'].to(config.general.device)
            # print(tokenizer.decode(input_ids[0]))
            attention_mask = batch['attention_mask'].to(config.general.device)
            start_positions = batch['start_positions'].to(config.general.device)
            end_positions = batch['end_positions'].to(config.general.device)
            # train model on batch and return outputs (incl. loss)
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
    return model

def evaluate_squad(val_dataset, model):
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size)
    acc = []
    # initialize loop for progress bar
    loop = tqdm(val_loader)
    # loop through batches
    for batch in loop:
        # we don't need to calculate gradients as we're not training
        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch['input_ids'].to(config.general.device)
            attention_mask = batch['attention_mask'].to(config.general.device)
            start_true = batch['start_positions'].to(config.general.device)
            end_true = batch['end_positions'].to(config.general.device)
            # make predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            # pull preds out
            start_pred = torch.argmax(outputs['start_logits'], dim=-1)
            end_pred = torch.argmax(outputs['end_logits'], dim=-1)
            acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
            acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
    print(sum(acc)/len(acc))
    return sum(acc)/len(acc)


def check_results(custom_model, initial_model, config, tokenizer, layers):        
    train_data, val_data = get_squad_dataset(config, tokenizer, custom_model)
        
    model = finetune(train_data, model)
    
    print('Validation evaluation:\n')
    a = evaluate_squad(val_data, model)
    
    cols = ['dataset_name', 'layers', 'accuracy']
    data = {i: '' for i in cols}
    data['layers'] = str(layers)
    data['dataset_name'] = dataset_name
    data['accuracy'] = a
    if not os.path.exists(f'{config.data.data_path}/results_{config.data.model_save_pattern}.csv'):
        results = pd.DataFrame([], columns = cols)
    else:
        results = pd.read_csv(f'{config.data.data_path}/results_{config.data.model_save_pattern}.csv', index_col = [0])
    results.loc[len(results)] = data
    print(f'{config.data.data_path}/results_{config.data.model_save_pattern}.csv')
    results.to_csv(f'{config.data.data_path}/results_{config.data.model_save_pattern}.csv')
        

def main(args):
    with open(args.config_path, "r") as f:
        config = ConfigWrapper(yaml.load(f, Loader=yaml.FullLoader))

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, max_length=config.general.max_len)
    initial_model = AutoModel.from_pretrained(config.model.model_name).to(config.general.device)
    eval_datasets = load_datasets(config.data.eval_datasets, config.data.cut_size)

    linear_model = linear_model = GPTWrapperLin(initial_model.transformer, LinearClassifierGPTAttention, config, layer_nums=config.attention_config.layers_to_change)


    if config.attention_config.split_heads or config.attention_config.model_for_each_head:
        init_linear_modules(config, linear_model)
    else:
        init_linear_modules2(config, linear_model)

    initial_model = initial_model.to(config.general.device)
    linear_model = linear_model.to(config.general.device)
    
    check_results(linear_model, initial_model, config, tokenizer, eval_datasets)

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("--config_path", default='config_gpt.yaml', help="path to your config (config.yaml default)")

    args=parser.parse_args()
    main(args)