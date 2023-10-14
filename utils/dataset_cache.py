import h5py
import os
import numpy as np
import torch

from .utils import hidden_to_heads
from tqdm.auto import tqdm
from copy import deepcopy

def cache_embeddings(config, train_datasets, tokenizer, initial_model, cache_embeddings):
    if not cache_embeddings:
        return
    dataset_names = list(train_datasets)
    for dataset_idx, dataset_name in enumerate(dataset_names):
        if not os.path.exists(f'{config.data.data_path}/{dataset_name}'):
            os.makedirs(f'{config.data.data_path}/{dataset_name}')
        for ex_idx, ex in tqdm(enumerate(train_datasets[dataset_name]['train'])):
            if ex_idx > config.data.cache_cut_size:
                continue
            field1, field2 = config.data.train_datasets_fields[dataset_idx]
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
            aa = initial_model(encoded_inputs, output_hidden_states=True, output_attentions=True)
        
            layers = config.data.layers if isinstance(config.data.layers, list) else range(config.model.layers)
            for layer in layers:
                if not os.path.exists(f'{config.data.data_path}/{dataset_name}/layer_{layer}'):
                    os.makedirs(f'{config.data.data_path}/{dataset_name}/layer_{layer}')
        
                current_hidden_states = aa.hidden_states[layer][0].detach().cpu().numpy()
                next_hidden_states = aa.hidden_states[layer + 1][0].detach().cpu().numpy()
                attentions = aa.attentions[layer][0].detach().cpu().numpy() # .item()
    
                with h5py.File(f'{config.data.data_path}/{dataset_name}/layer_{layer}/{ex_idx}.hdf5', 'w') as f:
                    f.create_dataset("current_hidden_states", data=current_hidden_states)
                    f.create_dataset("next_hidden_states", data=next_hidden_states)
                    f.create_dataset("attentions", data=attentions)


def cache_features_for_regression(config, train_datasets, tokenizer, initial_model, cache_embeddings=False, layers=[0], heads=[0], split_for_heads=False):
    X_train, y_train = [], []
    X_test, y_test = [], []
    dataset_names = list(train_datasets)
    for dataset_idx, dataset_name in enumerate(dataset_names):
        if not os.path.exists(f'{config.data.data_path}/linear_cached/{dataset_name}'):
            os.makedirs(f'{config.data.data_path}/linear_cached/{dataset_name}')
        for ex_idx, ex in tqdm(enumerate(train_datasets[dataset_name]['train']), total=len(train_datasets[dataset_name]['train'])):
            if ex_idx > config.data.cache_cut_size:
                continue

            field1, field2 = config.data.train_datasets_fields[dataset_idx]
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

            seq_len = encoded_inputs.shape[1]
            if not cache_embeddings:
                aa = initial_model(encoded_inputs, output_hidden_states=True, output_attentions=True)
            # print(seq_len)
            for layer in layers:
                if cache_embeddings:
                    with h5py.File(f'{config.data.data_path}/{dataset_name}/layer_{layer}/{ex_idx}.hdf5', 'r') as f:
                        current_hidden_states = f['current_hidden_states'][()]
                        next_hidden_states = f['next_hidden_states'][()]
                        attentions = f['attentions'][()]
                else:
                    current_hidden_states = aa.hidden_states[layer][0].detach().cpu().numpy()
                    next_hidden_states = aa.hidden_states[layer + 1][0].detach().cpu().numpy()
                    attentions = aa.attentions[layer][0].detach().cpu().numpy()

                if config.attention_config.split_heads:
                    # print(current_hidden_states.shape)
                    current_hidden_states = hidden_to_heads(torch.tensor(current_hidden_states), config).detach().cpu().numpy()
                    
                for from_ in range(seq_len):
                    if np.random.choice([0, 1], size=1, p=[1-config.data.prob_of_take, config.data.prob_of_take])[0] < 0.5:
                        continue
                    for to_ in range(seq_len):
                        if np.random.choice([0, 1], size=1, p=[1-config.data.prob_of_take, config.data.prob_of_take])[0] < 0.5:
                            continue
    
                        for head_num in heads:
                            if not os.path.exists(f'{config.data.data_path}/{dataset_name}/layer_{layer}/head_{head_num}'):
                                os.makedirs(f'{config.data.data_path}/{dataset_name}/layer_{layer}/head_{head_num}')
                            
                            chs_from = current_hidden_states[from_] if not config.attention_config.split_heads else current_hidden_states[from_][head_num]
                            chs_to = current_hidden_states[to_] if not config.attention_config.split_heads else current_hidden_states[to_][head_num]
                            full_data_to_linear = {
                                'hidden_from': chs_from, 
                                'hidden_to': chs_to, 
                                'pos_from': from_ / config.data.norm_len_factor,
                                'pos_to': to_ / config.data.norm_len_factor,
                                'relev_pos_from': (seq_len - from_) / config.data.norm_len_factor,
                                'relev_pos_to': (seq_len - to_) / config.data.norm_len_factor,
                                'inv_pos_from': (from_ / seq_len),
                                'inv_pos_to': (to_ / seq_len),
                                'inv_relev_pos_from': ((seq_len - from_) / seq_len),
                                'inv_relev_pos_to': ((seq_len - to_) / seq_len),
                                'seq_len': seq_len / config.data.norm_len_factor, 
                                'inv_seq_len': (1 / seq_len),
                                'target':  np.log(np.where(attentions[head_num][from_, to_] == 0,  1e-10, attentions[head_num][from_, to_])),
                                'head_num': head_num
                            }

                            if config.data.cache_train_features:
                                with h5py.File(f'{config.data.data_path}/{dataset_name}/layer_{layer}/head_{head_num}/{config.data.train_features_prefix}_{ex_idx}_{from_}_{to_}.hdf5', 'w') as f:
                                    for fn, fv in full_data_to_linear.items():
                                        f.create_dataset(f'{fn}', data=fv)    
    return


def get_dataset_for_regression(config, train_datasets, tokenizer, initial_model, cache_embeddings=False, layer=0, heads=[0], split_for_heads=False, ):
    X_train, y_train = [], []
    X_test, y_test = [], []
    dataset_names = list(train_datasets)
    for dataset_idx, dataset_name in enumerate(dataset_names):
        if not os.path.exists(f'{config.data.data_path}/linear_cached/{dataset_name}'):
            os.makedirs(f'{config.data.data_path}/linear_cached/{dataset_name}')
        for ex_idx, ex in tqdm(enumerate(train_datasets[dataset_name]['train']), total=len(train_datasets[dataset_name]['train'])):
            if ex_idx > config.data.cache_cut_size:
                continue

            field1, field2 = config.data.train_datasets_fields[dataset_idx]
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

            seq_len = encoded_inputs.shape[1]
            if not cache_embeddings:
                aa = initial_model(encoded_inputs, output_hidden_states=True, output_attentions=True)
            # print(seq_len)
            for from_ in range(seq_len):
                if np.random.choice([0, 1], size=1, p=[1-config.data.prob_of_take, config.data.prob_of_take])[0] < 0.5:
                    continue
                for to_ in range(seq_len):
                    if np.random.choice([0, 1], size=1, p=[1-config.data.prob_of_take, config.data.prob_of_take])[0] < 0.5:
                        continue
                    if cache_embeddings:
                        with h5py.File(f'{config.data.data_path}/{dataset_name}/layer_{layer}/{ex_idx}.hdf5', 'r') as f:
                            current_hidden_states = f['current_hidden_states'][()]
                            next_hidden_states = f['next_hidden_states'][()]
                            attentions = f['attentions'][()]
                    else:
                        current_hidden_states = aa.hidden_states[layer][0].detach().cpu().numpy()
                        next_hidden_states = aa.hidden_states[layer + 1][0].detach().cpu().numpy()
                        attentions = aa.attentions[layer][0].detach().cpu().numpy()

                    if config.attention_config.split_heads:
                        # print(current_hidden_states.shape)
                        current_hidden_states = hidden_to_heads(torch.tensor(current_hidden_states), config).detach().cpu().numpy()

                    for head_num in heads:
                        if not os.path.exists(f'{config.data.data_path}/{dataset_name}/layer_{layer}/head_{head_num}'):
                            os.makedirs(f'{config.data.data_path}/{dataset_name}/layer_{layer}/head_{head_num}')
                        
                        chs_from = current_hidden_states[from_] if not config.attention_config.split_heads else current_hidden_states[from_][head_num]
                        chs_to = current_hidden_states[to_] if not config.attention_config.split_heads else current_hidden_states[to_][head_num]
                        full_data_to_linear = {
                            'hidden_from': chs_from, 
                            'hidden_to': chs_to, 
                            'pos_from': from_ / config.data.norm_len_factor,
                            'pos_to': to_ / config.data.norm_len_factor,
                            'relev_pos_from': (seq_len - from_) / config.data.norm_len_factor,
                            'relev_pos_to': (seq_len - to_) / config.data.norm_len_factor,
                            'inv_pos_from': (from_ / seq_len),
                            'inv_pos_to': (to_ / seq_len),
                            'inv_relev_pos_from': ((seq_len - from_) / seq_len),
                            'inv_relev_pos_to': ((seq_len - to_) / seq_len),
                            'seq_len': seq_len / config.data.norm_len_factor, 
                            'inv_seq_len': (1 / seq_len),
                            'target': np.log(np.where(attentions[head_num][from_, to_] == 0,  1e-10, attentions[head_num][from_, to_])),
                            'head_num': head_nums
                        }
                        if config.data.cache_train_features:
                            with h5py.File(f'{config.data.data_path}/{dataset_name}/layer_{layer}/head_{head_num}/{config.data.train_features_prefix}_{ex_idx}_{from_}_{to_}.hdf5', 'w') as f:
                                for fn, fv in full_data_to_linear.items():
                                    f.create_dataset(f'{fn}', data=fv)
    
                        feature_vector = []
                        for feature_name in config.attention_config.features:
                            if isinstance(full_data_to_linear[feature_name], int) or \
                               isinstance(full_data_to_linear[feature_name], float):
                                feature_vector.append(full_data_to_linear[feature_name])
                            else:
                                feature_vector += list(full_data_to_linear[feature_name])
                                
                        X_train.append(feature_vector)
                        y_train.append(np.log(np.where(attentions[head_num][from_, to_] == 0, attentions[head_num][from_, to_],
                                                          attentions[head_num][from_, to_] + 1e-10)))

    l = round(len(X_train)*config.data.train_prop)
    X_test = np.array(X_train[l:])
    y_test = np.array(y_train[l:])
    X_train = np.array(X_train[:l])
    y_train = np.array(y_train[:l])
    if config.data.cache_train_dataset:
        if not os.path.exists(f'{config.data.data_path}/linear_cached/{layer}'):
            os.makedirs(f'{config.data.data_path}/linear_cached/{layer}')
        with open(f'{config.data.data_path}/linear_cached/{layer}/x_train.npy', 'wb') as f:
            np.save(f, X_train)
        with open(f'{config.data.data_path}/linear_cached/{layer}/x_test.npy', 'wb') as f:
            np.save(f, X_test)
        with open(f'{config.data.data_path}/linear_cached/{layer}/y_train.npy', 'wb') as f:
            np.save(f, y_train)
        with open(f'{config.data.data_path}/linear_cached/{layer}/y_test.npy', 'wb') as f:
            np.save(f, y_test)
        
    return X_train, y_train, X_test, y_test


def build_dict_dataset_from_cached(config, train_datasets, layer=0, heads=[0], features=None, split_hidden=False):
    assert config.data.cache_train_features == True, 'according to config - you did not cached features'
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    dataset_names = list(train_datasets)
        
    for dataset_idx, dataset_name in enumerate(dataset_names):
        for head_num in heads:
            if features is None:
                features = ['hidden_from', 'hidden_to', 'pos_from', 'pos_to', 'relev_pos_from',
                            'relev_pos_to', 'inv_pos_from', 'inv_pos_to', 'inv_relev_pos_from',
                            'inv_relev_pos_to', 'seq_len', 'inv_seq_len', 'head_num']
            full_data_to_linear = {}
            all_pathes = os.listdir(f'{config.data.data_path}/{dataset_name}/layer_{layer}/head_{head_num}')
            all_pathes = [el for el in all_pathes if config.data.train_features_prefix in el]
            for el in all_pathes:
                path_to_file = f'{config.data.data_path}/{dataset_name}/layer_{layer}/head_{head_num}/{el}'
                if os.path.exists(path_to_file):
                    with h5py.File(path_to_file, 'r') as f:
                        #print(f.keys())
                        for fn in features:
                            if 'hidden' in fn and split_hidden:
                                new_hidden_size = config.attention_config['d_model'] // config.attention_config['num_heads']
                                slice = (head_num * new_hidden_size, (head_num + 1) * new_hidden_size)
                                full_data_to_linear[fn] = f[fn][()][slice[0]:slice[1]]
                            else:
                                full_data_to_linear[fn] = f[fn][()]
                        tgt = f['target'][()] 

                    X_train.append(deepcopy(full_data_to_linear))
                    y_train.append(tgt) 

    l = round(len(X_train) * config.data.train_prop)
    X_test = np.array(X_train[l:])
    y_test = np.array(y_train[l:])
    X_train = np.array(X_train[:l])
    y_train = np.array(y_train[:l])        
    return X_train, y_train, X_test, y_test


def build_dataset_from_cached(config, train_datasets, layer=0, heads=[0]):
    assert config.data.cache_train_features == True, 'according to config - you did not cached features'
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    dataset_names = list(train_datasets)
    if not os.path.exists(f'{config.data.data_path}/linear_cached'):
        os.makedirs(f'{config.data.data_path}/linear_cached')
        
    for dataset_idx, dataset_name in enumerate(dataset_names):
       
        for head_num in heads:
            features = ['hidden_from', 'hidden_to', 'pos_from', 'pos_to', 'relev_pos_from',
                        'relev_pos_to', 'inv_pos_from', 'inv_pos_to', 'inv_relev_pos_from',
                        'inv_relev_pos_to', 'seq_len', 'inv_seq_len', 'head_num']
            full_data_to_linear = {}
            all_pathes = os.listdir(f'{config.data.data_path}/{dataset_name}/layer_{layer}/head_{head_num}')
            all_pathes = [el for el in all_pathes if config.data.train_features_prefix in el]
            for el in all_pathes:
                path_to_file = f'{config.data.data_path}/{dataset_name}/layer_{layer}/head_{head_num}/{el}'
                if os.path.exists(path_to_file):
                    try:
                        with h5py.File(path_to_file, 'r') as f:
                            #print(f.keys())
                            for fn in features:
                                full_data_to_linear[fn] = f[fn][()]
                            tgt = f['target'][()]
                    except:
                        continue
                    #print(full_data_to_linear)
                    feature_vector = []
                    for feature_name in config.attention_config.features:
                        if isinstance(full_data_to_linear[feature_name], int) or \
                           isinstance(full_data_to_linear[feature_name], float) or \
                           isinstance(full_data_to_linear[feature_name], np.int64) or \
                           isinstance(full_data_to_linear[feature_name], np.float32):
                            feature_vector.append(full_data_to_linear[feature_name])
                        else:
                            feature_vector += list(full_data_to_linear[feature_name])
                            
                    X_train.append(feature_vector)
                    y_train.append(tgt) 

    l = round(len(X_train) * config.data.train_prop)
    X_test = np.array(X_train[l:])
    y_test = np.array(y_train[l:])
    X_train = np.array(X_train[:l])
    y_train = np.array(y_train[:l])
    if config.data.cache_train_dataset:
        if not os.path.exists(f'{config.data.data_path}/linear_cached/{layer}'):
            os.makedirs(f'{config.data.data_path}/linear_cached/{layer}')
        with open(f'{config.data.data_path}/linear_cached/{layer}/x_train.npy', 'wb') as f:
            np.save(f, X_train)
        with open(f'{config.data.data_path}/linear_cached/{layer}/x_test.npy', 'wb') as f:
            np.save(f, X_test)
        with open(f'{config.data.data_path}/linear_cached/{layer}/y_train.npy', 'wb') as f:
            np.save(f, y_train)
        with open(f'{config.data.data_path}/linear_cached/{layer}/y_test.npy', 'wb') as f:
            np.save(f, y_test)
        
    return X_train, y_train, X_test, y_test

def load_cached_dataset(config, layer=0):
    assert config.data.cache_train_dataset == True, 'according to config - you did not cached dataset'

    if config.data.cache_train_dataset:
        if not os.path.exists(f'{config.data.data_path}/linear_cached/{layer}'):
            os.makedirs(f'{config.data.data_path}/linear_cached/{layer}')
        with open(f'{config.data.data_path}/linear_cached/{layer}/x_train.npy', 'rb') as f:
            X_train = np.load(f)
        with open(f'{config.data.data_path}/linear_cached/{layer}/x_test.npy', 'rb') as f:
            X_test = np.load(f)
        with open(f'{config.data.data_path}/linear_cached/{layer}/y_train.npy', 'rb') as f:
            y_train = np.load(f)
        with open(f'{config.data.data_path}/linear_cached/{layer}/y_test.npy', 'rb') as f:
            y_test = np.load(f)

    return X_train, y_train, X_test, y_test