from typing import Tuple, List, Dict, Optional, Union
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

def load_datasets(benchmark_name: str, dataset_names: List[str], cut_len=None):
    ans = {}
    for name in dataset_names:
        dataset = load_dataset(benchmark_name, name)
        ans[f'{name}'] = dataset
    if cut_len is not None:
        ans = cut_datasets(ans, cut_len)
    return ans

def cut_datasets(
    datasets: Dict[str, Union[DatasetDict, Dict[str, Dataset], Dict[str, Dataset]]], 
    cut_len:int = 100
):
    for dataset_name, dataset in datasets.items():
        for dataset_split_name, dataset_split in dataset.items():
            new_split = dataset_split[:cut_len]
            if isinstance(new_split, dict):
                new_split = [dict(zip(new_split,t)) for t in zip(*new_split.values())]
            datasets[dataset_name][dataset_split_name] = new_split
    return datasets
        