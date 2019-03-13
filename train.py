import os
import sys
import tqdm
import time
import json
import logging
import argparse
from sgdr import CosineWithRestarts
from model_utils import create_train_model
from utils import calculate_cmvn, convert_to
from datasets import EMPHASISDataset, collate_fn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

with open('./hparams.json', 'r') as f:
    hparams = json.load(f)

logger = logging.getLogger(__name__)


def train_one_acoustic_epoch(train_loader, model, device, optimizer):
    model.train()
    tr_loss = 0.0
    num_steps = 0

    pbar = tqdm(train_loader, total=(len(train_loader)), unit=' batches')
    for b, (input_batch, target_batch, mask, uv_mask) in enumerate(
            pbar):
        input = input_batch.to(device=device)
        target = target_batch.to(device=device)
        mask = mask.to(device=device)
        uv_mask = uv_mask.to(device=device)
        target = torch.cat((target[:, :, :hparams['spec_units'] + hparams['lf0_units']],
                            target[:, :, -(hparams['energy_units'] + hparams['cap_units']):]), -1)
        uv_target = target[:, :, -1]
        uv_target[uv_target > 0.5] = 1
        uv_target = uv_target.long()

        output, uv_output = model(input)

        # mask the loss
        output *= mask
        uv_output *= uv_mask

        output_loss = F.mse_loss(output, target)
        uv_output = uv_output.view(-1, 2)
        uv_target = uv_target.view(-1, 1)
        uv_output_loss = F.cross_entropy(uv_output, uv_target.squeeze())
        loss = output_loss + uv_output_loss
        tr_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_steps += 1
    return tr_loss / num_steps

def eval_one_acoustic_epoch(valid_loader, model, device):
    model.eval()
    val_loss = 0.0
    num_steps = 0

    pbar = tqdm(valid_loader, total=(len(valid_loader)), unit=' batches')
    for b, (input_batch, target_batch, mask, uv_mask) in enumerate(
            pbar):
        input = input_batch.to(device=device)

        target = target_batch.to(device=device)
        target = torch.cat([target[:, :, :hparams['spec_units'] + hparams['lf0_units']],
                           target[:, :, -(hparams['energy_units'] + hparams['bap_units']):]], -1)
        uv_target = target[:, :, -1]

        output, uv_output = model(input)

        # mask the loss
        output *= mask
        uv_output *= uv_mask

        output_loss = F.mse_loss(output, target)

        uv_output = uv_output.view(-1, 2)
        uv_target = uv_target.view(-1, 1)
        uv_output_loss = F.cross_entropy(uv_output, uv_target)
        loss = output_loss + uv_output_loss
        val_loss += loss
        num_steps += 1
    model.train()
    return val_loss / num_steps

def train_one_duration_epoch(train_loader, model, device, optimizer):
    model.train()
    tr_loss = 0.0
    num_steps = 0

    pbar = tqdm(train_loader, total=(len(train_loader)), unit=' batches')
    for b, (input_batch, target_batch, mask) in enumerate(
            pbar):
        input = input_batch.to(device=device)
        target = target_batch.to(device=device)

        output = model(input)
        # mask the loss
        output *= mask
        output_loss = F.mse_loss(output, target)
        loss = output_loss
        tr_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_steps += 1
    return tr_loss / num_steps

def eval_one_duration_epoch(valid_loader, model, device):
    model.eval()
    val_loss = 0.0
    num_steps = 0

    pbar = tqdm(valid_loader, total=(len(valid_loader)), unit=' batches')
    for b, (input_batch, target_batch, mask) in enumerate(
            pbar):
        input = input_batch.to(device=device)
        target = target_batch.to(device=device)

        output = model(input)
        # mask the loss
        output *= mask
        output_loss = F.mse_loss(output, target)
        loss = output_loss
        val_loss += loss
        num_steps += 1
    model.train()
    return val_loss / num_steps

def train_model(args, model_type, model, optimizer, lr_scheduler, exp_name, device, epoch, checkpoint_path):
    data_path = os.path.join(args.base_dir, args.data)
    train_dataset = EMPHASISDataset(f'{data_path}/train', f'./config/train.lst')
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hparams['batch_size'], sampler=train_sampler,
                        num_workers=6, collate_fn=collate_fn, pin_memory=False)

    valid_dataset = EMPHASISDataset(f'{data_path}/vaild', f'./config/valid.lst')
    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), sampler=valid_sampler,
                              num_workers=6, collate_fn=collate_fn, pin_memory=False)

    for cur_epoch in tqdm(range(epoch, hparams[f'max_{model_type}_epochs'])):
        # train one epoch
        time_start = time.time()
        lr_scheduler.step(cur_epoch)
        lr = lr_scheduler.get_lr()[0]
        if model_type == 'acoustic':
            tr_loss = train_one_acoustic_epoch(train_loader, model, device, optimizer)
        else:
            tr_loss = train_one_duration_epoch(train_loader, model, device, optimizer)
        time_end = time.time()
        used_time = time_end - time_start

        # validate one epoch
        if model_type == 'acoustic':
            val_loss = eval_one_acoustic_epoch(valid_loader, model, device)
        else:
            val_loss = eval_one_duration_epoch(valid_loader, model, device)

        logger.info(f'EPOCH {cur_epoch}: TRAIN AVG.LOSS {tr_loss:.4f}, learning_rate: {lr:g}'
                    f'CROSSVAL AVG.LOSS {val_loss:.4f}, TIME USED {used_time:.2f}')

        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }

        torch.save(state, f'{checkpoint_path}/{exp_name}_epoch{cur_epoch}_lrate{lr:g}_tr{tr_loss:.4f}_cv{val_loss:g}.tar')
        logger.info(
            f'save state to {checkpoint_path}/{exp_name}_epoch{cur_epoch}_lrate{lr:g}_tr{tr_loss:.4f}_cv{val_loss:g}.tar succeed')

        # add a blank line for log readability
        print()
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--data', default='./data/', type=str,
                        help='path to dataset contains inputs and targets')
    parser.add_argument('--log_dir', default='EMPHASIS', type=str, help='path to save checkpoint')
    parser.add_argument('--restore_from', default=None, type=str,
                        help='the checkpoint such as xxx.tar restored from the log_dir you set')
    parser.add_argument('--name', default='EMPHASIS', type=str,
                        help='name of the experiment')

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', level=logging.INFO,
                        stream=sys.stdout)


    epoch = 0
    model_type = hparams['model_type']
    exp_name = args.name
    model = create_train_model(model_type)
    device = torch.device('cuda:' + hparams['gpu_ids'] if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    checkpoint_path = os.path.join(args.log_dir, "checkpoint")
    optimizer = optim.Adam(model.parameters(), lr=hparams['initial_lr'], weight_decay=hparams['weight_decay'])
    lr_scheduler = CosineWithRestarts(optimizer, t_max=hparams[f'max_{model_type}_epochs'], eta_min=hparams['min_lr'])

    if args.restore_from is not None:
        # load the checkpoint ...
        cpkt_path = os.path.join(checkpoint_path, args.restore_from)
        if os.path.exists(cpkt_path):
            logger.info(f'loading checkpoint from {cpkt_path} ...')

            checkpoint = torch.load(cpkt_path)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            exp_name = cpkt_path.split('/')[-1].split("_")[0]
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            logger.info(f'loaded checkpoint from {cpkt_path} succeed')
        else:
            logger.error(f'checkpoint path:{checkpoint_path} does\'t exist!')


    train_model(args, model_type, model, optimizer,
                lr_scheduler, exp_name, device, epoch, checkpoint_path)

if __name__ == '__main__':
    main()