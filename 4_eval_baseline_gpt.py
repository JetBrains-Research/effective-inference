import os

import argparse
import yaml
import h5py
import numpy as np
import torch
import json
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import time
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer
from utils.dataset_cache import cache_embeddings, get_dataset_for_regression, build_dataset_from_cached, load_cached_dataset
from utils.dataset_cache import build_dict_dataset_from_cached
from utils.prepare_dataset import load_datasets, cut_datasets
from utils.config import ConfigWrapper
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, List, Dict, Optional, Union
from numpy.random import shuffle
from sklearn.metrics import r2_score
from random import choices
from datasets import Dataset

from IPython.display import clear_output
from collections import defaultdict

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer, GPT2ForQuestionAnswering
from transformers import GPT2TokenizerFast
from utils.attentions.bert.linear import BertWrapperLin, LinearClassifierBertAttention, LinearAttention
from utils.attentions.gpt2.gpt2_module_linear import GPTWrapperLin, LinearClassifierGPTAttention, LinearAttention
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from utils.attentions.gpt2.gpt2_modules import GPT2Wrapper, MeanGPT2Attention, WindowGPT2Attention, WindowMEANGPT2Attention
from transformers import GPT2ForQuestionAnswering
import random
from utils.dataset_utils import get_dict_batch, prepare_batches
from utils.train_linear_utils import train_epoch, eval_epoch, plot_history

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report
from itertools import combinations

tqdm_pbar = lambda x, y: tqdm(x, leave=True, position=0, total=len(x), desc=f'{y}')
def read_squad(path, n=2000):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    # initialize lists for contexts, questions, and answers
    contexts = []
    answers = []
    
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            t = ''
            t+=context+'\n\n'
            if len(passage['qas'])<3:
                continue
            for i, qa in enumerate(passage['qas']):
                question = qa['question']
                t += 'Q: ' + question + '\n'
                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'
                ans = qa[access]
                if i < 2:
                    t += 'A: ' + ans[0]['text'] + '\n\n'
                elif i==2:
                    t += 'A:'
                    contexts.append(t)
                    answers.append([i['text'] for i in ans])
                    if len(contexts)==n:
                        return contexts, answers
                    else: continue
                else: pass
              
    # return formatted data lists
    return contexts, answers



def add_end_token_to_question(input_dict):
    input_dict['question'] += special_tokens['bos_token']
    return input_dict


def divide_tokenized_text(tokenized_text_dict, block_size):
    concatenated_examples = {k: sum(tokenized_text_dict[k], []) for k in tokenized_text_dict.keys()}
    total_length = len(concatenated_examples[list(tokenized_text_dict.keys())[0]])
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    result['labels'] = result['input_ids'].copy()
    return result

tqdm_pbar = lambda x, y: tqdm(x, leave=True, position=0, total=len(x), desc=f'{y}')


def eval_perplexity(model, tokenizer, eval_dataset):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    
    training_args = TrainingArguments(
        f'./gpt2-squad',
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False, # Change to True to push the model to the Hub
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    eval_results = trainer.evaluate()
    return math.exp(eval_results["eval_loss"])
        


def check_results(custom_model, initial_model, eval_dataset, config, tokenizer, model_name,  layers):        
    
    p = eval_perplexity(custom_model, tokenizer, eval_dataset)
    
    cols = ['dataset_name', 'model_name', 'layers', 'perplexity', 'test_len', 'n_layers']
    data = {i: '' for i in cols}
    data['layers'] = str(layers)
    data['dataset_name'] = config.data.eval_datasets

    data['perplexity'] = p
    data['model_name'] = model_name
    data['test_len'] = len(eval_dataset)
    data['n_layers'] = len(layers)
    if not os.path.exists(f'{config.data.data_path}/baseline_{config.data.model_save_pattern}.csv'):
        results = pd.DataFrame([], columns = cols)
    else:
        results = pd.read_csv(f'{config.data.data_path}/baseline_{config.data.model_save_pattern}.csv', index_col = [0])
    results.loc[len(results)] = data
    print(data)
    results.to_csv(f'{config.data.data_path}/baseline_{config.data.model_save_pattern}.csv')
      

def main(args):
    
    
    with open(args.config_path, "r") as f:
        config = ConfigWrapper(yaml.load(f, Loader=yaml.FullLoader))
    print(config.data.data_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(config.model.model_name, max_length=config.general.max_len)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    special_tokens = tokenizer.special_tokens_map
    
    initial_model = GPT2LMHeadModel.from_pretrained(config.model.model_name).to(config.general.device)
    custom_model = GPT2LMHeadModel.from_pretrained(config.model.model_name).to(config.general.device)

    val_contexts, val_answers = read_squad('squad/train-v2.0.json', config.data.cut_size)

    dataset = Dataset.from_dict({'question': val_contexts})
    # dataset = dataset.map(add_end_token_to_question)
    tokenized_dataset = dataset.map(lambda input_dict: tokenizer(input_dict['question'], truncation=True), batched=True, num_proc=4, remove_columns=['question'])
    
    eval_dataset = tokenized_dataset.map(
        lambda tokenized_text_dict: divide_tokenized_text(tokenized_text_dict, block_size=512),
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    
    try:
        results = pd.read_csv(f'{config.data.data_path}/baseline_{config.data.model_save_pattern}.csv', index_col = [0])
        uniq = results['layers'].unique()
    except: uniq=[]
    lst= []
    for i in range(5, 11):
        if i > 1 and i < 11:
            comb = [x for x in list(map(list, list(combinations(list(range(1,12)), i)))) if str(x) not in uniq]
        else:
            comb = list(map(list, list(combinations(list(range(12)), i))))
        lst+= comb
        
    for i in tqdm(lst):
        layers_to_change = i
        print(layers_to_change)
        
        custom_model.transformer = GPT2Wrapper(initial_model.transformer, WindowGPT2Attention, i)
        custom_model = custom_model.to(config.general.device)
        check_results(custom_model, initial_model, eval_dataset, config, tokenizer, 'Window', layers_to_change)
    
    
        
if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("--config_path", default='config_gpt.yaml', help="path to your config (config.yaml default)")

    args=parser.parse_args()
    main(args)