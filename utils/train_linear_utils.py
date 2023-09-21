from tqdm.auto import tqdm
import torch
import torch.nn as nn
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .dataset_utils import prepare_batches
from IPython.display import clear_output
from .attentions.bert.linear import BertWrapperLin, LinearClassifierBertAttention, LinearAttention
from sklearn.metrics import r2_score

def train_epoch(model, optimizer, loss_fn, X_train, y_train, config, scheduler=None, use_pbar=False):
    loss_log = []
    all_results = []
    model.train()

    pbar = prepare_batches(X_train, y_train, config.attention_config.train_batch_size, config.general.device)
    if use_pbar:
        add_ = 0 if len(X_train) % config.attention_config.train_batch_size == 0 else 1
        total_len = (len(X_train) // config.attention_config.train_batch_size) + add_
        pbar = tqdm(pbar, total=total_len, position=0, leave=True)
    for batch_num, batch in enumerate(pbar):
        x_batch, y_batch = batch
        
        optimizer.zero_grad()
        
        result = model(**x_batch)
        loss = loss_fn(result, y_batch)

        loss.backward()
        optimizer.step()

        loss = loss.item()
        loss_log.append(loss)
        all_results.append(result)
        if scheduler is not None:
            scheduler.step()
    return loss_log, torch.cat(all_results).detach().cpu().numpy()

def eval_epoch(model, loss_fn, X_test, y_test, config, use_pbar=False):
    loss_log = []
    all_results = []
    model.eval()

    pbar = prepare_batches(X_test, y_test, config.attention_config.train_batch_size, config.general.device)
    if use_pbar:
        add_ = 0 if len(X_test) % config.attention_config.train_batch_size == 0 else 1
        total_len = (len(X_test) // config.attention_config.train_batch_size) + add_
        pbar = tqdm(pbar, total=total_len, position=0, leave=True)
    for batch_num, batch in enumerate(pbar):
        x_batch, y_batch = batch
        with torch.no_grad():
            result = model(**x_batch)
            loss = loss_fn(result, y_batch)    
            loss = loss.item()
            loss_log.append(loss)
            all_results.append(result)
    return loss_log, torch.cat(all_results).detach().cpu().numpy()

def plot_history(train_history, val_history, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)    
    points = np.array(val_history)
    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val', zorder=2)
    plt.xlabel('train steps')
    plt.legend(loc='best')
    plt.grid()
    plt.show()