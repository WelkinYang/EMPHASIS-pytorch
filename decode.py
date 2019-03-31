import os
import sys
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
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_' + args.name)
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config_' + args.name)
    data_list = open(os.path.join(config_dir, 'test.lst'), 'r').readlines()
    cmvn = np.load(os.path.join(data_dir, "train_cmvn.npz"))
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if args.model_type == 'acoustic':
        for input_name in data_list:
            input_name = input_name.split(' ')[0] + '.lab'
            logging.info(f'decode {input_name} ...')
            input = read_binary_file(os.path.join(os.path.join(data_dir, 'test', 'label'), input_name),
                                     dimension=hparams['in_channels'])
            input = torch.from_numpy(input).to(device)
            input = input.unsqueeze(0)
            output, uv_output = model(input)
            output = output.squeeze()
            uv_output = F.softmax(uv_output, dim=-1)[:, :, 0]
            uv_output = uv_output.squeeze()
            uv = torch.ones(uv_output.shape).to(device)
            uv[uv_output > 0.5] = 0.0
            uv = uv.unsqueeze(-1)
            output = torch.cat((uv, output), -1)
            output = output.cpu().squeeze().detach().numpy()
            uv = uv.cpu().squeeze().detach().numpy()
            output = output * cmvn['stddev_labels'] + cmvn["mean_labels"]

            cap = output[:, 1:hparams['cap_units']]
            sp = np.concatenate((output[:, hparams['cap_units'] + hparams['energy_units'] + 1:
                                           hparams['cap_units'] + hparams['energy_units'] + hparams['spec_units'] + 1],
                                 output[:,
                                 hparams['cap_units'] + 1:hparams['cap_units'] + hparams['energy_units'] + 1]), axis=-1)
            lf0 = output[:, hparams['cap_units'] + hparams['energy_units'] + hparams['spec_units'] + 1:
                            hparams['cap_units'] + hparams['energy_units'] + hparams['spec_units'] + hparams[
                                'lf0_units'] + 1]
            lf0[uv == 0] = -1.0e+10
            write_binary_file(sp, os.path.join(args.output, os.path.splitext(input_name)[0] + '.sp'), dtype=np.float64)
            write_binary_file(lf0, os.path.join(args.output, os.path.splitext(input_name)[0] + '.lf0'),
                              dtype=np.float32)
            write_binary_file(cap, os.path.join(args.output, os.path.splitext(input_name)[0] + '.ap'), dtype=np.float64)
    elif args.model_type == 'acoustic_mgc':
        for input_name in data_list:
            input_name = input_name.split(' ')[0] + '.lab'
            logging.info(f'decode {input_name} ...')
            input = read_binary_file(os.path.join(os.path.join(data_dir, 'test', 'label'), input_name),
                                     dimension=hparams['in_channels'])
            input = torch.from_numpy(input).to(device)
            input = input.unsqueeze(0)
            output, uv_output = model(input)
            output = output.squeeze()
            uv_output = F.softmax(uv_output, dim=-1)[:, :, 0]
            uv_output = uv_output.squeeze()
            uv = torch.ones(uv_output.shape).to(device)
            uv[uv_output > 0.5] = 0.0
            uv = uv.unsqueeze(-1)
            output = torch.cat((output[:, :hparams['mgc_units']],
                                uv, output[:, -(hparams['bap_units'] + hparams['lf0_units']):]), -1)
            output = output.cpu().squeeze().detach().numpy()
            uv = uv.cpu().squeeze().detach().numpy()
            output = output * cmvn['stddev_labels'] + cmvn["mean_labels"]

            mgc = output[:, :hparams['mgc_units']]
            lf0 = output[:, hparams['mgc_units'] + 1:hparams['mgc_units'] + hparams['lf0_units'] + 1]
            bap = output[:, -(hparams['bap_units']):]
            write_binary_file(mgc, os.path.join(args.output, os.path.splitext(input_name)[0] + '.mgc'))
            write_binary_file(lf0, os.path.join(args.output, os.path.splitext(input_name)[0] + '.lf0'))
            write_binary_file(bap, os.path.join(args.output, os.path.splitext(input_name)[0] + '.bap'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output', default='./test_cmp/', type=str,
                        help='path to output cmp')
    parser.add_argument('--model_type', default='')
    parser.add_argument('--name', default='')
    parser.add_argument('--use_cuda', default=False)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', level=logging.DEBUG,
                        stream=sys.stdout)

    model = create_train_model(args.model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() >= 1:
        model = nn.DataParallel(model)
    model.to(device)

    if args.use_cuda:
        checkpoint = torch.load(args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'])

    decode(args, model, device)


if __name__ == '__main__':
    main()
