import torch
from torch.utils.data import Dataset

import json
import numpy as np
import pandas as pd
from utils import read_binary_file


with open('./hparams.json', 'r') as f:
    hparams = json.load(f)

class EMPHASISDataset(Dataset):
    def __init__(self, path, id_path, model_type, sort=True):
        super(EMPHASISDataset, self).__init__()
        self.path = path
        self.meta_data = pd.read_csv(f'{id_path}', sep=' ',
                                     names=['id', 'label_dir', 'cmp_dir'],
                                     usecols=['id'],
                                     dtype={'id':str, 'label_dir':str, 'cmp_dir':str},
                                     index_col=False)

        self.meta_data.dropna(inplace=True)
        self.model_type = model_type

    def __getitem__(self, index):
        id = self.meta_data.iloc[index]['id']
        input = read_binary_file(f'{self.path}/label/{id}.lab', dimension=hparams['in_channels'])
        target = read_binary_file(f'{self.path}/cmp/{id}.cmp', dimension=hparams['mgc_target_channels']
        if self.model_type == 'acoustic_mgc' else hparams['target_channels'], dtype=np.float64)
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

    channels = targets[0].shape[2]

    mask = np.stack(_pad_mask(input_len, max_input_len, channels) for input_len in input_lens)
    uv_mask = np.stack(_pad_uv_mask(input_len, max_input_len) for input_len in input_lens)
    input_batch = np.stack(_pad_input(input, max_input_len) for input in inputs)
    target_batch = np.stack(_pad_target(target, max_target_len, channels) for target in targets)
    return torch.DoubleTensor(input_batch), torch.DoubleTensor(target_batch), torch.DoubleTensor(mask), torch.DoubleTensor(uv_mask)

def _pad_mask(len, max_len, channels):
    return np.concatenate([np.ones((len, channels-1)), np.zeros((max_len-len, channels-1))], axis=0)

def _pad_uv_mask(len, max_len):
    return np.concatenate([np.ones((len, hparams['uv_units'])), np.zeros((max_len-len, hparams['uv_units']))], axis=0)

def _pad_input(input, max_input_len):
    padded = np.zeros((max_input_len - len(input), hparams['in_channels'])) + hparams['acoustic_input_padded']
    return np.concatenate([input, padded], axis=0).astype(np.float32)

def _pad_target(target, max_target_len, channels):
    if hparams['model_type'] == 'acoustic' or 'acoustic_mgc':
        padded = np.zeros(max_target_len - len(target), channels) + \
                 hparams['acoustic_target_padded']
    else:
        padded = np.zeros(max_target_len - len(target)) + \
                 hparams['duration_target_padded']
    return np.concatenate([target, padded], axis=0)