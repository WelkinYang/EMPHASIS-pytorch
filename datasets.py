import torch
from torch.utils.data import Dataset

import json
import numpy as np
import pandas as pd
from utils import read_binary_file


with open('./hparams.json', 'r') as f:
    hparams = json.load(f)

class EMPHASISDataset(Dataset):
    def __init__(self, path, sort=True):
        super(EMPHASISDataset, self).__init__()
        self.path = path
        self.meta_data = pd.read_csv(f'{path}/metadata.csv', sep='|',
                                     names=['id', 'frame_nums'],
                                     index_col=False)

        self.meta_data.dropna(inplace=True)

        if sort:
            self.meta_data.sort_values(by=['frame_nums'], inplace=True)

    def __getitem__(self, index):
        id = self.meta_data.iloc[index]['id']
        input = read_binary_file(f'{path}/label/{id}.lab', dimension=hparams['in_channels'])
        target = read_binary_file(f'{path}/cmp/{id}.cmp', dimension=hparams['target_channels'])
        return input, target

    def __len__(self):
        return len(self.meta_data)

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    input_lens = [len(input) for input in inputs]
    target_lens = [len(target) for target in targets]

    max_input_len = max(input_lens)
    max_target_len = max(target_lens)

    mask = np.stack(_pad_mask(input_len, max_input_len) for input_len in input_lens)
    input_batch = np.stack(_pad_input(input, max_input_len) for input in inputs)
    target_batch = np.stack(_pad_target(input, max_target_len) for target in targets)
    return torch.FloatTensor(input_batch), torch.FloatTensor(target_batch), torch.IntTensor(mask)

def _pad_mask(len, max_len):
    return np.concatenate([np.ones(len), np.zeros(max_len-len)], axis=0)

def _pad_input(input, max_input_len):
    padded = np.zeros(max_input_len - len(input), hparams['acoustic_in_channels']) + hparams['acoustic_input_padded']
    return np.concatenate([input, padded], axis=0)

def _pad_target(target, max_target_len):
    if hparams['model_type'] == 'acoustic':
        padded = np.zeros(max_target_len - len(target), hparams['target_channels']) + \
                 hparams['acoustic_target_padded']
    else:
        padded = np.zeros(max_target_len - len(target)) + \
                 hparams['duration_target_padded']
    return np.concatenate([target, padded], axis=0)