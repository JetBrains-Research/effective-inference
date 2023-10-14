import os
import argparse

import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import json
import seaborn as sns
import matplotlib.pyplot as plt

from utils.dataset_cache import cache_embeddings, get_dataset_for_regression, build_dataset_from_cached, load_cached_dataset
from utils.dataset_cache import build_dict_dataset_from_cached
from utils.prepare_dataset import load_datasets, cut_datasets
from utils.config import ConfigWrapper
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, List, Dict, Optional, Union
from numpy.random import shuffle
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score, max_error, mean_absolute_percentage_error

from utils.attentions.bert.linear import BertWrapperLin, LinearClassifierBertAttention, LinearAttention
from utils.dataset_utils import get_dict_batch, prepare_batches
from utils.train_linear_utils import train_epoch, eval_epoch, plot_history
import pandas as pd


def train_linear_model(X_train, X_test, y_train, y_test, config,layer, head, save_pattern='', 
                       use_plots=False, save_final_results=True, 
                       verbose=False, use_pbars=False, save_model=True):
    
    add_ = 0 if len(X_train) % config.attention_config.train_batch_size == 0 else 1
    total_len = (len(X_train) // config.attention_config.train_batch_size) + add_
    
    model = LinearAttention(config.attention_config).to(config.general.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=total_len, epochs=config.general.num_epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_log = []
    val_log = []
    for epoch in range(config.general.num_epochs):
        if use_plots:
            clear_output()
            
        train_loss, _ = train_epoch(model, optimizer, criterion, X_train, y_train, config, scheduler=scheduler, use_pbar=use_pbars)
        val_loss, val_preds = eval_epoch(model, criterion, X_test, y_test, config, use_pbar=use_pbars)
        train_log.extend(train_loss)
        steps = len(train_loss)
        val_log.append((steps * (epoch + 1), np.mean(val_loss)))
        
        if use_plots:
            print(f'{epoch} -- VAL R2 score:', r2_score(y_test, val_preds))
            plot_history(train_log, val_log)
        elif verbose:
            print(f'{epoch} -- Mean train loss:', np.mean(train_loss))
            print(f'{epoch} -- Mean val loss:', np.mean(val_loss))
            print(f'{epoch} -- VAL R2 score:', r2_score(y_test, val_preds))
            print()

        if epoch + 1 == config.general.num_epochs and save_final_results and save_pattern != '':
            if not os.path.exists(f'{config.data.data_path}/linear_models'):
                os.makedirs(f'{config.data.data_path}/linear_models')
            if not os.path.exists(f'{config.data.data_path}/linear_models/{save_pattern}'):
                os.makedirs(f'{config.data.data_path}/linear_models/{save_pattern}')
            with open(f'{config.data.data_path}/linear_models/{save_pattern}/preds.json', 'wb') as f:
                np.save(f, val_preds) # json.dump(val_preds, f)
            with open(f'{config.data.data_path}/linear_models/{save_pattern}/true.json', 'wb') as f:
                np.save(f, y_test) # json.dump(y_test, f)

    if save_model:
        if not os.path.exists(f'{config.data.data_path}/linear_models'):
            os.makedirs(f'{config.data.data_path}/linear_models')
        if not os.path.exists(f'{config.data.data_path}/linear_models/{save_pattern}'):
            os.makedirs(f'{config.data.data_path}/linear_models/{save_pattern}')
        model.to('cpu')
        torch.save(model.state_dict(), f'{config.data.data_path}/linear_models/{save_pattern}/model.pth')

    if epoch + 1 == config.general.num_epochs and not verbose and not use_plots:
        print(f'Final val loss:', np.mean(val_loss))
        print(f'Final val R2 score:', r2_score(y_test, val_preds))
        
    cols = ['dataset_name','layer', 'head', 'model_name', 'r2', 'mse', 'explained_variance_score', 'max_error', 
                                     'mean_absolute_percentage_error']
    data = {i: '' for i in cols}
    data['layer'] = layer
    data['head'] = head
    data['dataset_name'] = str(config.data.train_datasets)
    data['r2'] = r2_score(y_test, val_preds)
    data['mse'] = mean_squared_error(y_test, val_preds)
    data['explained_variance_score'] = explained_variance_score(y_test, val_preds)
    data['max_error'] = max_error(y_test, val_preds)
    data['mean_absolute_percentage_error'] = mean_absolute_percentage_error(y_test, val_preds)
    data['model_name'] = str(model)
    
    if not os.path.exists(f'{config.data.data_path}/metrics_{config.data.model_save_pattern}.csv'):
        metrics = pd.DataFrame([], columns = cols)
    else:
        metrics = pd.read_csv(f'{config.data.data_path}/metrics_{config.data.model_save_pattern}.csv', index_col = [0])
    metrics.loc[len(metrics)] = data
    metrics.to_csv(f'{config.data.data_path}/metrics_{config.data.model_save_pattern}.csv')
    
    
    return model


def main(args):
    with open(args.config_path, "r") as f:
        config = ConfigWrapper(yaml.load(f, Loader=yaml.FullLoader))

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, max_length=config.general.max_len)
    initial_model = AutoModel.from_pretrained(config.model.model_name).to(config.general.device)
    train_datasets = load_datasets(config.data.train_datasets, config.data.cut_size)
    
    cols = ['dataset_name','layer', 'head', 'model_name', 'r2', 'mse', 'explained_variance_score', 'max_error', 
                                     'mean_absolute_percentage_error']
    metrics = pd.DataFrame([], columns = cols)
    metrics.to_csv(f'{config.data.data_path}/{config.data.model_save_pattern}.csv')
    
    if config.attention_config.split_heads or config.attention_config.model_for_each_head:
        pbar = tqdm(total=len(config.attention_config.layers_to_train) * len(config.attention_config.heads_to_train), position=0, leave=True)
        for layer_N in config.attention_config.layers_to_train:
            for head_N in config.attention_config.heads_to_train:
                print(f'Training {layer_N} layer, {head_N} head')
                X_train, y_train, X_test, y_test = build_dict_dataset_from_cached(config, train_datasets, layer=layer_N, heads=[head_N], 
                                                                          features=config.attention_config.features, 
                                                                          split_hidden=config.attention_config.split_heads_in_data)
                X_train = X_train[:len(X_train)//2]
                y_train = y_train[:len(X_train)//2]
                X_test = X_test[:len(X_test)//2]
                y_test = y_test[:len(X_test)//2]
                print('Train size:', len(X_train))
                print(X_train[10]['hidden_to'].shape)
                train_linear_model(X_train, X_test, y_train, y_test, config,layer=layer_N, head=head_N,
                                   save_pattern=f'{config.data.model_save_pattern}_{layer_N}_{head_N}', 
                                   use_plots=False, save_final_results=args.save_final_results, 
                                   verbose=args.verbose, use_pbars=args.use_pbars, save_model=args.save_final_models)
                pbar.update(1)
    
    else:
        pbar = tqdm(total=12, position=0, leave=True)
        for layer_N in config.attention_config.layers_to_train:
            X_train, y_train, X_test, y_test = build_dict_dataset_from_cached(config, train_datasets, layer=layer_N,
                                                                              heads=config.attention_config.heads_to_train,
                                                                              features=config.attention_config.features, split_hidden=False)
            X_train = X_train[:len(X_train)//2]
            y_train = y_train[:len(X_train)//2]
            X_test = X_test[:len(X_test)//2]
            y_test = y_test[:len(X_test)//2]
            print('Train size:', len(X_train))
            train_linear_model(X_train, X_test, y_train, y_test, config,layer=layer_N, head=None,
                               save_pattern=f'{config.data.model_save_pattern}_{layer_N}', 
                               use_plots=False, save_final_results=args.save_final_results, 
                               verbose=args.verbose, use_pbars=args.use_pbars, save_model=args.save_final_models)
            pbar.update(1)

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("--config_path", default='config.yaml', help="path to your config (config.yaml default)")
    parser.add_argument("--verbose", action="store_true", help="Verbose training")
    parser.add_argument("--use_pbars", action="store_true",
                        help="Use pbar while training")
    parser.add_argument("--save_final_results", action="store_true",
                        help="Save final results?")
    parser.add_argument("--save_final_models", action="store_true",
                        help="save_final_models")

    args=parser.parse_args()
    main(args)