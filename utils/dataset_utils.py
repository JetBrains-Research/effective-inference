import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

def get_dict_batch(samples_arr, device):
    sample = samples_arr[0]
    final = {}
    for k, _ in sample.items():
        batched_feature = []
        for el in samples_arr:
            if 'hidden' in k:
                batched_feature.append(torch.tensor(el[k]))
            else:
                batched_feature.append(el[k])

        if 'hidden' in k:
            batched_feature = torch.stack(batched_feature)
        else:
            batched_feature = torch.tensor(batched_feature)
        final[k] = batched_feature.to(device)
    return final
    
def prepare_batches(dataset_X, dataset_y, n, device):
    shuffle_idx = np.arange(len(dataset_X)//2)
    np.random.shuffle(shuffle_idx)
    new_dataset = dataset_X[shuffle_idx]
    new_dataset_y = dataset_y[shuffle_idx]
    # looping till length l
    for i in range(0, len(new_dataset), n): 
        yield get_dict_batch(new_dataset[i:i + n], device), torch.tensor(new_dataset_y[i:i + n]).view(-1, 1).to(device)