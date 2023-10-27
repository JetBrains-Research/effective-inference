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
import time
from transformers import GPT2LMHeadModel
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


tqdm_pbar = lambda x, y: tqdm(x, leave=True, position=0, total=len(x), desc=f'{y}')

import torch.nn.functional as F

def generate_text(model, tokenizer, config, prompt, max_length=50):
    model = model.to(config.general.device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(config.general.device)
    input_ids_0 = input_ids
    answer = ''

    log_prob_sum = 0

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model.transformer(input_ids)
            logits = model.lm_head(outputs.last_hidden_state)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=-1).to(config.general.device)
            # print(tokenizer.decode(next_token, skip_special_tokens=True), end = ' ')
            answer += tokenizer.decode(next_token, skip_special_tokens=True) + ' '
            log_prob_sum += torch.log_softmax(logits, dim=-1)[:, -1, next_token].item()
    
    # Compute perplexity
    perplexity = torch.exp(torch.tensor(-log_prob_sum / (max_length + 1)))
    return answer, perplexity
    
import time

def eval_accuracy(model, tokenizer, config, val_dataset, true_answers):
    # print(model.transformer)
    pred_answers, perplexities = [],[]
    s = time.time()
    results = list(map(lambda x: generate_text(model.eval(), tokenizer, config, x, 5), val_dataset))
    pred_answers, perplexities = map(list, zip(*results))

    # as_list, bs_list = zip(*[generate_text(model.eval(), tokenizer, config, x, 5) for x in val_dataset])
    # for x in tqdm(val_dataset):
    #     gen, perplexity = generate_text(model.eval(), tokenizer, config, x, 5)
    #     pred_answers.append(gen)
    #     perplexities.append(perplexity)
    # print(time.time() - s)
    
    accuracy = 0
    for i in tqdm(range(len(pred_answers))):
        pred = pred_answers[i]
        real = true_answers[i]
        for answer in real:
            if answer in pred: 
                accuracy+=1
                break
    return accuracy/len(pred_answers), sum(perplexities)/len(perplexities)
        


def check_results(custom_model, initial_model, val_contexts, val_answers, config, tokenizer, model_name,  layers):        
    
    a, p = eval_accuracy(custom_model, tokenizer, config,  val_contexts, val_answers)
    
    cols = ['dataset_name', 'model_name', 'layers', 'accuracy', 'perplexity', 'test_len', 'n_layers']
    data = {i: '' for i in cols}
    data['layers'] = str(layers)
    data['dataset_name'] = config.data.eval_datasets
    data['accuracy'] = a
    data['perplexity'] = p
    data['model_name'] = model_name
    data['test_len'] = len(val_contexts)
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
    
    initial_model = GPT2LMHeadModel.from_pretrained(config.model.model_name).to(config.general.device)
    custom_model = GPT2LMHeadModel.from_pretrained(config.model.model_name).to(config.general.device)

    val_contexts, val_answers = read_squad('squad/train-v2.0.json', config.data.cut_size)
    try:
        results = pd.read_csv(f'{config.data.data_path}/baseline_{config.data.model_save_pattern}.csv', index_col = [0])
        uniq = results['layers'].unique()
    except: uniq=[]
    lst= []
    for i in range(0, 11):
        if i > 1 and i < 11:
            comb = random.choices([x for x in list(map(list, list(combinations(list(range(12)), i)))) if str(x) not in uniq], k=5) 
        else:
            comb = list(map(list, list(combinations(list(range(12)), i))))
        lst+= comb
        
    for i in tqdm(lst):
        layers_to_change = i
        print(layers_to_change)
        
        custom_model.transformer = GPT2Wrapper(initial_model.transformer, WindowGPT2Attention, i)
        custom_model = custom_model.to(config.general.device)
        check_results(custom_model, initial_model, val_contexts, val_answers, config, tokenizer, 'Uniform', layers_to_change)
    
    
        
if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("--config_path", default='config_gpt.yaml', help="path to your config (config.yaml default)")

    args=parser.parse_args()
    main(args)