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

tqdm_pbar = lambda x, y: tqdm(x, leave=True, position=0, total=len(x), desc=f'{y}')
def get_cls_embeddings_for_dataset(dataset_idx, dataset_name, dataset, config, tokenizer, model, eval_datasets,
                                   pbar_func=tqdm_pbar):
    collected_embeddings = defaultdict(list)

    for split, data in eval_datasets[dataset_name].items():
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


def init_linear_modules(config, linear_model):
    for layer_num, bert_att in enumerate(linear_model.bert_model.encoder.layer):
        for param_name, param in bert_att.named_modules():
            if '.' in param_name and 'linear_model' in param_name.split('.')[-1]:
                head_num = int(param_name.split('_')[-1])
                save_pattern = f"{config.data.model_save_pattern}_{layer_num}_{head_num}"
                param.load_state_dict(torch.load(f'{config.data.data_path}/linear_models/{save_pattern}/model.pth'), strict=False)

def init_linear_modules2(config, linear_model):
    for layer_num, bert_att in enumerate(linear_model.bert_model.encoder.layer):
        for param_name, param in bert_att.named_modules():
            if '.' in param_name and 'linear_model' in param_name.split('.')[-1]:
                save_pattern = f"{config.data.model_save_pattern}_{layer_num}"
                param.load_state_dict(torch.load(f'{config.data.data_path}/linear_models/{save_pattern}/model.pth'), strict=False)

def check_results(custom_model, initial_model, datasets, config, tokenizer, eval_datasets):
    for dataset_idx, (dataset_name, dataset) in enumerate(datasets.items()):
        print(f"{dataset_name}\n")

        # print('Original')

        # dataset_embeddings_orig = get_cls_embeddings_for_dataset(
        #     dataset_idx, 
        #     dataset_name,
        #     dataset, 
        #     config,
        #     tokenizer, 
        #     initial_model,
        #     eval_datasets)
        
        # train_dataset_embeddings = torch.cat(dataset_embeddings_orig['train'], dim=0)
        # valid_dataset_embeddings = torch.cat(dataset_embeddings_orig['validation'], dim=0)
        # test_dataset_embeddings = torch.cat(dataset_embeddings_orig['test'], dim=0)
        
        # classif = train_linear(train_dataset_embeddings.cpu(), [el['label'] for el in dataset['train']])
        # valid_preds = evaluate_classifier(classif, valid_dataset_embeddings.cpu())
        # print('Validation evaluation:\n')
        # get_metrics_report([el['label'] for el in dataset['validation']], valid_preds)
        # # print(train_dataset_embeddings.shape)

        
        print('\nLinear:')
        
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
        test_dataset_embeddings = torch.cat(dataset_embeddings_custom['test'], dim=0)


        classif = train_linear(train_dataset_embeddings.cpu(), [el['label'] for el in dataset['train']])
        valid_preds = evaluate_classifier(classif, valid_dataset_embeddings.cpu())
        print('Validation evaluation:\n')
        f, a = get_metrics_report([el['label'] for el in dataset['validation']], valid_preds)
        
        cols = ['dataset_name', 'layers', 'accuracy', 'f1']
        data = {i: '' for i in cols}
        data['layers'] = str(config.attention_config.layers_to_change)
        data['dataset_name'] = dataset_name
        data['accuracy'] = a
        data['f1'] = f
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

    linear_model = BertWrapperLin(initial_model, LinearClassifierBertAttention, config, layer_nums=config.attention_config.layers_to_change)

    if config.attention_config.split_heads or config.attention_config.model_for_each_head:
        init_linear_modules(config, linear_model)
    else:
        init_linear_modules2(config, linear_model)

    initial_model = initial_model.to(config.general.device)
    linear_model = linear_model.to(config.general.device)
    
    check_results(linear_model, initial_model, eval_datasets, config, tokenizer, eval_datasets)

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("--config_path", default='config.yaml', help="path to your config (config.yaml default)")

    args=parser.parse_args()
    main(args)