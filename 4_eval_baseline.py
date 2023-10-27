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

from utils.attentions.bert.linear import BertWrapperLin, LinearClassifierBertAttention, LinearAttention
from utils.attentions.bert.window import WindowBert2Attention, BertWrapper, BertWrapperBoth
from utils.attentions.bert.uniform import UniformBert2Attention, BertWrapperUniform
from utils.dataset_utils import get_dict_batch, prepare_batches
from utils.train_linear_utils import train_epoch, eval_epoch, plot_history

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report
from itertools import combinations

tqdm_pbar = lambda x, y: tqdm(x, leave=True, position=0, total=len(x), desc=f'{y}')
def get_cls_embeddings_for_dataset(dataset_idx, dataset_name, dataset, config, tokenizer, model, eval_datasets,
                                   pbar_func=tqdm_pbar):
    collected_embeddings = defaultdict(list)

    for split, data in eval_datasets[dataset_name].items():
        if split == 'test':
            continue
        pbar = pbar_func(list(enumerate(data)), f"{split} {dataset_name}") if pbar_func is not None else data
        for ex_idx, ex in pbar:
            field1, field2 = config.data.eval_datasets_fields[dataset_idx]
            if field2 != '':
                encoded_inputs = tokenizer.encode(
                                ex[field1],
                                ex[field2],
                                truncation=True,
                                return_tensors='pt'
                            ).to(config.general.device)
            else:
                encoded_inputs = tokenizer.encode(
                                ex[field1],
                                truncation=True,
                                return_tensors='pt'
                            ).to(config.general.device)
            
            with torch.no_grad():
                outputs = model(encoded_inputs)

            # Get the embedding of the [CLS] token
            cls_embedding = outputs.last_hidden_state[:, 0, :]

            # Append the [CLS] embedding to the list
            collected_embeddings[split].append(cls_embedding)
    return collected_embeddings

def train_linear(X_train, y_train):
    classifier = LogisticRegression(solver='lbfgs', max_iter=3000)
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_classifier(classifier, X, y=None):
    predictions = classifier.predict(X)
    return predictions

def get_metrics_report(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred,  average='weighted')
    print('Weighted F1', f1)
    print('Accuracy', accuracy)
    print('-------------------------------')
    return f1, accuracy



def check_results(custom_model, initial_model, datasets, config, tokenizer, eval_datasets, model_name, layers_w = [], layers_u = []):
    for dataset_idx, (dataset_name, dataset) in enumerate(datasets.items()):
        print(f"{dataset_name}\n")

        print(':')
        
        dataset_embeddings_custom = get_cls_embeddings_for_dataset(
            dataset_idx, 
            dataset_name,
            dataset, 
            config,
            tokenizer, 
            custom_model,
            eval_datasets)
        
        train_dataset_embeddings = torch.cat(dataset_embeddings_custom['train'], dim=0)
        valid_dataset_embeddings = torch.cat(dataset_embeddings_custom['validation'], dim=0)
        # test_dataset_embeddings = torch.cat(dataset_embeddings_custom['test'], dim=0)


        classif = train_linear(train_dataset_embeddings.cpu(), [el['label'] for el in dataset['train']])
        valid_preds = evaluate_classifier(classif, valid_dataset_embeddings.cpu())
        print('Validation evaluation:\n')
        f, a = get_metrics_report([el['label'] for el in dataset['validation']], valid_preds)   
        cols = ['model_name', 'dataset_name', 'layers_window', 'layers_uniform', 'n_layers', 'accuracy', 'f1', 'train_len', 'val_len', 'test_len']
        data = {i: '' for i in cols}
        data['model_name'] = model_name
        data['layers_window'] = str(layers_w)
        data['layers_uniform'] = str(layers_u)
        data['dataset_name'] = dataset_name
        data['accuracy'] = a
        data['f1'] = f
        data['n_layers'] = len(layers_w)+len(layers_u)
        data['train_len'] = len(train_dataset_embeddings)
        data['val_len'] = len(valid_dataset_embeddings)
        data['test_len'] = 1000
        print(data)
        if not os.path.exists(f'{config.data.data_path}/baseline_{config.data.model_save_pattern}.csv'):
            results = pd.DataFrame([], columns = cols)
        else:
            results = pd.read_csv(f'{config.data.data_path}/baseline_{config.data.model_save_pattern}.csv', index_col = [0])
        results.loc[len(results)] = data
        results.to_csv(f'{config.data.data_path}/baseline_{config.data.model_save_pattern}.csv')
        

def main(args):
    
    with open(args.config_path, "r") as f:
        config = ConfigWrapper(yaml.load(f, Loader=yaml.FullLoader))

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, max_length=config.general.max_len)
    initial_model = AutoModel.from_pretrained(config.model.model_name).to(config.general.device)
    eval_datasets = load_datasets(config.data.eval_datasets, config.data.cut_size)

    # initial_model = initial_model.to(config.general.device)
    # check_results(initial_model, initial_model, eval_datasets, config, tokenizer, eval_datasets, model_name='Original')
    lst_w = [[0, 3, 7], [0, 7, 9], [1, 3, 10], [0, 4, 11], [0, 2, 3], [0, 3, 9], [0, 2, 5], [0, 1, 7], [0, 1, 3], [0, 1, 2, 3, 9]]
    lst_u = [[0, 2, 8], [3, 8, 11], [0, 3, 8, 11], [2, 8, 10], [0, 9, 10], [0, 3, 9], [8, 9, 10], [0, 1, 8], [2, 8, 9], [2, 8, 9]]
    for i in tqdm(range(len(lst_w))):
        print(lst_w[i], lst_u[i])
        custom_model = BertWrapperBoth(initial_model, lst_w[i], lst_u[i])
        # initial_model = initial_model.to(config.general.device)
        custom_model = custom_model.to(config.general.device)
        check_results(custom_model, initial_model, eval_datasets, config, tokenizer, eval_datasets, 'Both', lst_w[i], lst_u[i])
    
    # custom_model = BertWrapperUniform(initial_model, UniformBert2Attention, config.attention_config.layers_to_change)
    # initial_model = initial_model.to(config.general.device)
    # custom_model = custom_model.to(config.general.device)
    # check_results(custom_model, initial_model, eval_datasets, config, tokenizer, eval_datasets, 'Uniform')
    # print(time.time() - start)
        
if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("--config_path", default='config.yaml', help="path to your config (config.yaml default)")

    args=parser.parse_args()
    main(args)