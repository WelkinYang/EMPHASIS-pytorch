import os
import tqdm
import json
import logging
import argparse
import numpy as np
from model_utils import create_train_model
from datasets import EMPHASISDataset
from utils import read_binary_file, write_binary_file

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

with open('./hparams.json', 'r') as f:
    hparams = json.load(f)

def decode(args, model, device):
    model.eval()
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    data_list = os.path.join(data_dir, 'test')
    cmvn = np.load(os.path.join(data_dir, "train_cmvn.npz"))
    for input_name in data_list:
        logging.info(f'decode {input_name} ...')
        input = read_binary_file(os.path.join(data_list, input_name),
                         dimension=hparams['in_channels'])
        output = model(input)
        output = output * cmvn['stddev_labels'] + cmvn["mean_labels"]
        write_binary_file(os.path.join(args.output, os.path.splitext(input_name) + '.cmp'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output', default='./test_cmp/', type=str,
                        help='path to output cmp')

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', level=logging.DEBUG,
                        stream=sys.stdout)

    model_type = hparams['model_type']
    model = create_train_model(model_type)

    os.environ["CUDA_VISIBEL_DEVICES"] = hparams['gpu_ids']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])

    decode(args, model, device)

if __name__ == '__main__':
    main()