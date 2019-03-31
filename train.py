import os
import sys
import time
import json
import logging
import argparse
from tqdm import tqdm
from model_utils import create_train_model
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
        uv_target = target[:, :, 0]
        uv_target[uv_target > 0.5] = 1
        uv_target[uv_target <= 0.5] = 0
        uv_target = uv_target.long()
        target = target[:, :, 1:]

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
    val_loss = 0.0
    num_steps = 0

    pbar = tqdm(valid_loader, total=(len(valid_loader)), unit=' batches')
    for b, (input_batch, target_batch, mask, uv_mask) in enumerate(
            pbar):
        input = input_batch.to(device=device)
        target = target_batch.to(device=device)
        mask = mask.to(device=device)
        uv_mask = uv_mask.to(device=device)
        uv_target = target[:, :, 0]
        uv_target[uv_target > 0.5] = 1
        uv_target[uv_target <= 0.5] = 0
        uv_target = uv_target.long()

        target = target[:, :, 1:]

        output, uv_output = model(input)

        # mask the loss
        output *= mask
        uv_output *= uv_mask

        output_loss = F.mse_loss(output, target)

        uv_output = uv_output.view(-1, 2)
        uv_target = uv_target.view(-1, 1)
        uv_output_loss = F.cross_entropy(uv_output, uv_target.squeeze())
        loss = output_loss + uv_output_loss
        val_loss += loss.item()
        num_steps += 1
    return val_loss / num_steps


def train_one_acoustic_mgc_epoch(train_loader, model, device, optimizer):
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

        uv_target = target[:, :, hparams['mgc_units']:
                                 hparams['mgc_units'] + 1]

        target = torch.cat((target[:, :, :hparams['mgc_units']],
                            target[:, :, -(hparams['bap_units'] + hparams['lf0_units']):]), -1)
        uv_target[uv_target >= 0.5] = 1
        uv_target[uv_target < 0.5] = 0
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


def eval_one_acoustic_mgc_epoch(valid_loader, model, device):
    val_loss = 0.0
    num_steps = 0

    pbar = tqdm(valid_loader, total=(len(valid_loader)), unit=' batches')
    for b, (input_batch, target_batch, mask, uv_mask) in enumerate(
            pbar):
        input = input_batch.to(device=device)
        target = target_batch.to(device=device)
        mask = mask.to(device=device)
        uv_mask = uv_mask.to(device=device)
        uv_mask = uv_mask.to(device=device)

        uv_target = target[:, :, hparams['mgc_units']:
                                 hparams['mgc_units'] + 1]
        target = torch.cat((target[:, :, :hparams['mgc_units']],
                            target[:, :, -(hparams['bap_units'] + hparams['lf0_units']):]), -1)

        uv_target[uv_target >= 0.5] = 1
        uv_target[uv_target < 0.5] = 0
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
        val_loss += loss.item()
        num_steps += 1
    return val_loss / num_steps


def train_one_acoustic_dcbhg_mgc_epoch(train_loader, model, device, optimizer):
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

        uv_target = target[:, :, hparams['mgc_units']:
                                 hparams['mgc_units'] + 1]

        lf0_target = target[:, :, -1].unsqueeze(-1)

        target = torch.cat((target[:, :, :hparams['mgc_units']],
                            target[:, :, -(hparams['bap_units'] + hparams['lf0_units']):]), -1)
        uv_target[uv_target >= 0.5] = 1
        uv_target[uv_target < 0.5] = 0
        uv_target = uv_target.long()

        output, uv_output = model(input, lf0_target)
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


def eval_one_acoustic_dcbhg_mgc_epoch(valid_loader, model, device):
    val_loss = 0.0
    num_steps = 0

    pbar = tqdm(valid_loader, total=(len(valid_loader)), unit=' batches')
    for b, (input_batch, target_batch, mask, uv_mask) in enumerate(
            pbar):
        input = input_batch.to(device=device)
        target = target_batch.to(device=device)
        mask = mask.to(device=device)
        uv_mask = uv_mask.to(device=device)
        uv_mask = uv_mask.to(device=device)

        uv_target = target[:, :, hparams['mgc_units']:
                                 hparams['mgc_units'] + 1]
        lf0_target = target[:, :, -1].unsqueeze(-1)

        target = torch.cat((target[:, :, :hparams['mgc_units']],
                            target[:, :, -(hparams['bap_units'] + hparams['lf0_units']):]), -1)

        uv_target[uv_target >= 0.5] = 1
        uv_target[uv_target < 0.5] = 0
        uv_target = uv_target.long()


        output, uv_output = model(input, lf0_target)

        # mask the loss
        output *= mask
        uv_output *= uv_mask

        output_loss = F.mse_loss(output, target)

        uv_output = uv_output.view(-1, 2)
        uv_target = uv_target.view(-1, 1)
        uv_output_loss = F.cross_entropy(uv_output, uv_target.squeeze())
        loss = output_loss + uv_output_loss
        val_loss += loss.item()
        num_steps += 1
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


def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def train_model(args, model_type, model, optimizer, lr_scheduler, exp_name, device, epoch, checkpoint_path):
    data_path = os.path.join(args.base_dir, args.data)
    train_dataset = EMPHASISDataset(f'{data_path}/train', f'./config_{exp_name}/train.lst', model_type)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hparams['batch_size'], sampler=train_sampler,
                              num_workers=6, collate_fn=collate_fn, pin_memory=False)

    valid_dataset = EMPHASISDataset(f'{data_path}/valid', f'./config_{exp_name}/valid.lst', model_type)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=hparams['batch_size'], sampler=valid_sampler,
                              num_workers=6, collate_fn=collate_fn, pin_memory=False)
    prev_val_loss = 1000.0
    prev_checkpoint_path = '.'

    for cur_epoch in tqdm(range(epoch, hparams['max_epochs'])):
        # train one epoch
        time_start = time.time()
        if model_type == 'acoustic':
            tr_loss = train_one_acoustic_epoch(train_loader, model, device, optimizer)
        elif model_type == 'acoustic_mgc':
            tr_loss = train_one_acoustic_mgc_epoch(train_loader, model, device, optimizer)
        elif model_type == 'acoustic_dcbhg_mgc':
            tr_loss = train_one_acoustic_dcbhg_mgc_epoch(train_loader, model, device, optimizer)
        else:
            tr_loss = train_one_duration_epoch(train_loader, model, device, optimizer)
        time_end = time.time()
        used_time = time_end - time_start

        # validate one epoch
        if model_type == 'acoustic':
            val_loss = eval_one_acoustic_epoch(valid_loader, model, device)
        elif model_type == 'acoustic_mgc':
            val_loss = eval_one_acoustic_mgc_epoch(valid_loader, model, device)
        elif model_type == 'acoustic_dcbhg_mgc':
            val_loss = eval_one_acoustic_dcbhg_mgc_epoch(valid_loader, model, device)
        else:
            val_loss = eval_one_duration_epoch(valid_loader, model, device)

        lr_scheduler.step(val_loss)
        lr = get_lr(optimizer)

        logger.info(f'EPOCH {cur_epoch}: TRAIN AVG.LOSS {tr_loss:.4f}, learning_rate: {lr:g} '
                    f'CROSSVAL AVG.LOSS {val_loss:.4f}, TIME USED {used_time:.2f}')

        if val_loss >= prev_val_loss:
            logger.info(f'The CROSSVAL AVG.LOSS does\'nt reduce, so we need to reload the last checkpoint')
            checkpoint = torch.load(prev_checkpoint_path)
            cur_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])

            logger.info(f'Loaded checkpoint from {prev_checkpoint_path} succeed')
        else:
            state = {
                'epoch': cur_epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }

            torch.save(state,
                       f'{checkpoint_path}/{exp_name}_epoch{cur_epoch}_lrate{lr:g}_tr{tr_loss:.4f}_cv{val_loss:g}.tar')
            logger.info(
                f'Save state to {checkpoint_path}/{exp_name}_epoch{cur_epoch}_lrate{lr:g}_tr{tr_loss:.4f}_cv{val_loss:g}.tar succeed')
            prev_val_loss = val_loss
            prev_checkpoint_path = f'{checkpoint_path}/{exp_name}_epoch{cur_epoch}_lrate{lr:g}_tr{tr_loss:.4f}_cv{val_loss:g}.tar'

        # add a blank line for log readability
        print()
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--data', default='data', type=str,
                        help='path to dataset contains inputs and targets')
    parser.add_argument('--log_dir', default='EMPHASIS', type=str, help='path to save checkpoint')
    parser.add_argument('--restore_from', default=None, type=str,
                        help='the checkpoint such as xxx.tar restored from the log_dir you set')
    parser.add_argument('--model_type', default='acoustic', type=str,
                        help='model type which is either acoustic or acoustic_mgc')
    parser.add_argument('--name', default='EMPHASIS', type=str,
                        help='name of the experiment')

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', level=logging.INFO,
                        stream=sys.stdout)

    epoch = 0
    model_type = args.model_type
    exp_name = args.name
    model = create_train_model(model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() >= 1:
        model = nn.DataParallel(model)
    model.to(device)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    checkpoint_path = os.path.join(args.log_dir, "checkpoint")
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    optimizer = optim.Adam(model.parameters(), lr=hparams['initial_lr'], weight_decay=hparams['weight_decay'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                                        min_lr=hparams['min_lr'])

    if args.restore_from is not None:
        # load the checkpoint ...
        cpkt_path = os.path.join(checkpoint_path, args.restore_from)
        if os.path.exists(cpkt_path):
            logger.info(f'Loading checkpoint from {cpkt_path} ...')

            checkpoint = torch.load(cpkt_path)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            exp_name = cpkt_path.split('/')[-1].split("_epoch")[0]
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            logger.info(f'Loaded checkpoint from {cpkt_path} succeed')
        else:
            logger.error(f'Checkpoint path:{checkpoint_path} does\'t exist!')

    train_model(args, model_type, model, optimizer,
                lr_scheduler, exp_name, device, epoch, checkpoint_path)


if __name__ == '__main__':
    main()
