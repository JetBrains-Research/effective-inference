import yaml
import os
import h5py
import numpy as np
import torch
import argparse, sys

from utils.prepare_dataset import load_datasets, cut_datasets
from utils.dataset_cache import cache_embeddings, cache_features_for_regression, build_dataset_from_cached, load_cached_dataset
from utils.config import ConfigWrapper
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from numpy.random import shuffle




def main(args):
    with open(args.config_path, "r") as f:
        config = ConfigWrapper(yaml.load(f, Loader=yaml.FullLoader))

    if not os.path.exists(config.data.data_path):
        os.makedirs(config.data.data_path)
        print(f'Data directory {config.data.data_path} created')
    else:
        print('This directory already exists')

    train_datasets = load_datasets(config.data.train_datasets, config.data.cut_size)

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, max_length=config.general.max_len)
    initial_model = AutoModel.from_pretrained(config.model.model_name).to(config.general.device)

    cache_embeddings(config, train_datasets, tokenizer, initial_model, cache_embeddings=args.cache_embeddings)

    if args.prepare_dataset_for_regression:
        layers = config.data.layers if isinstance(config.data.layers, list) else range(config.model.layers_num)
        heads = config.data.heads if isinstance(config.data.heads, list) else list(range(config.model.heads_num))
        cache_features_for_regression(config, train_datasets, tokenizer, initial_model, layers=layers, heads=heads, cache_embeddings=args.cache_embeddings)

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("--config_path", default='config.yaml', help="path to your config (config.yaml default)")
    parser.add_argument("--cache_embeddings", action="store_true", help="Cache embeddings")
    parser.add_argument("--prepare_dataset_for_regression", action="store_true",
                        help="prepare dataset for regression")

    args=parser.parse_args()
    main(args)