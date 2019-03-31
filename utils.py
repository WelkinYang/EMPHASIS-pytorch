import os
import sys
import json
import struct
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

with open('./hparams.json', 'r') as f:
    hparams = json.load(f)

logger = logging.getLogger(__name__)


def pad(inputs, padding):
    return F.pad(inputs, (padding // 2, padding // 2 if padding % 2 == 0 else int(padding / 2 + 1)))


def Conv1d(inputs, conv, is_training, batch_norm=None, activation=None, padding=None):
    # the Conv1d of pytroch chanages the channels at the 1 dim
    # [batch_size, max_time, feature_dims] -> [batch_size, feature_dims, max_time]
    inputs = torch.transpose(inputs, 1, 2)
    if padding is not None:
        inputs = pad(inputs, padding)

    conv1d_output = conv(inputs)
    if batch_norm is not None:
        batch_norm_output = batch_norm(conv1d_output)
        batch_norm_output = torch.transpose(batch_norm_output, 1, 2)
    else:
        batch_norm_output = torch.transpose(conv1d_output, 1, 2)
    if activation is not None:
        batch_norm_output = activation(batch_norm_output)
    return F.dropout(batch_norm_output, p=hparams["dropout_rate"], training=is_training)


def MaxPool1d(inputs, maxpool, padding=None):
    if padding is not None:
        inputs = pad(inputs, padding)
    outputs = maxpool(inputs)
    return outputs


def highwaynet(inputs, activation, units=128):
    H = F.linear(inputs, weight=torch.nn.init.normal_(torch.empty(units, inputs.size(2))).cuda())
    H = activation[0](H)
    T = F.linear(inputs, weight=torch.nn.init.normal_(torch.empty(units, inputs.size(2)).cuda()),
                 bias=nn.init.constant_(torch.empty(1, 1, units), -0.1).cuda())
    T = activation[1](T)
    return H * T + inputs * (1.0 - T)


class HighwayNet(nn.Module):
    def __init__(self, activation=None, units=128):
        super(HighwayNet, self).__init__()

        self.activation = activation
        self.H = nn.Linear(units, units)
        self.T = nn.Linear(units, units)
        torch.nn.init.constant_(self.T.bias, val=-1.0)

    def forward(self, input):
        H_output = self.H(input)
        if self.activation[0] is not None:
            H_output = self.activation[0](H_output)

        T_output = self.T(H_output)
        if self.activation[1] is not None:
            T_output = self.activation[1](T_output)

        return H_output * T_output + input * (1.0 - T_output)


def calculate_cmvn(name, config_dir, output_dir, model_type):
    """Calculate mean and var."""
    logger.info("Calculating mean and var of %s" % name)
    config_filename = open(os.path.join(config_dir, name + '.lst'))

    inputs_frame_count, labels_frame_count = 0, 0
    for line in config_filename:
        utt_id, inputs_path, labels_path = line.strip().split()
        logger.info("Reading utterance %s" % utt_id)
        inputs = read_binary_file(inputs_path, hparams['in_channels'])
        labels = read_binary_file(labels_path, hparams['target_channels'] if model_type == 'acoustic' else
        hparams['mgc_target_channels'], dtype=np.float64 if model_type == 'acoustic'
        else np.float32)
        if inputs_frame_count == 0:  # create numpy array for accumulating
            ex_inputs = np.sum(inputs, axis=0)
            ex2_inputs = np.sum(inputs ** 2, axis=0)
            ex_labels = np.sum(labels, axis=0)
            ex2_labels = np.sum(labels ** 2, axis=0)
        else:
            ex_inputs += np.sum(inputs, axis=0)
            ex2_inputs += np.sum(inputs ** 2, axis=0)
            ex_labels += np.sum(labels, axis=0)
            ex2_labels += np.sum(labels ** 2, axis=0)
        inputs_frame_count += len(inputs)
        labels_frame_count += len(labels)

    mean_inputs = ex_inputs / inputs_frame_count
    stddev_inputs = np.sqrt(np.abs(ex2_inputs / inputs_frame_count - mean_inputs ** 2))
    stddev_inputs[stddev_inputs < 1e-20] = 1e-20

    mean_labels = ex_labels / labels_frame_count
    stddev_labels = np.sqrt(np.abs(ex2_labels / labels_frame_count - mean_labels ** 2))
    stddev_labels[stddev_labels < 1e-20] = 1e-20

    if model_type == 'acoustic':
        mean_labels[0] = 0.0
        stddev_labels[0] = 1.0
    elif model_type == 'acoustic_mgc':
        mean_labels[60] = 0.0
        stddev_labels[60] = 1.0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cmvn_name = os.path.join(output_dir, name + "_cmvn.npz")
    np.savez(cmvn_name,
             mean_inputs=mean_inputs,
             stddev_inputs=stddev_inputs,
             mean_labels=mean_labels,
             stddev_labels=stddev_labels)
    config_filename.close()
    logger.info("Wrote to %s" % cmvn_name)


def convert_to(name, config_dir, output_dir, model_type, apply_cmvn=True):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(os.path.join(output_dir, name)):
        os.mkdir(os.path.join(output_dir, name))
    if not os.path.exists(os.path.join(output_dir, name, 'label')):
        os.mkdir(os.path.join(output_dir, name, 'label'))
    if not os.path.exists(os.path.join(output_dir, name, 'cmp')):
        os.mkdir(os.path.join(output_dir, name, 'cmp'))
    cmvn = np.load(os.path.join(output_dir, "train_cmvn.npz"))
    config_file = open(config_dir + ".lst")
    for line in config_file:
        if name != 'test':
            utt_id, inputs_path, labels_path = line.strip().split()
            inputs_outdir = os.path.join(output_dir, name, 'label', f'{utt_id}.lab')
            labels_outdir = os.path.join(output_dir, name, 'cmp', f'{utt_id}.cmp')
        else:
            utt_id, inputs_path = line.strip().split()
            inputs_outdir = os.path.join(output_dir, name, 'label', f'{utt_id}.lab')

        logger.info(f'Writing utterance {utt_id} ...')
        inputs = read_binary_file(inputs_path, hparams['in_channels']).astype(np.float32)
        if name != 'test':
            labels = read_binary_file(labels_path, hparams['target_channels'] if model_type == 'acoustic' else
            hparams['mgc_target_channels'], dtype=np.float64 if model_type == 'acoustic'
            else np.float32).astype(np.float64 if model_type == 'acoustic' else np.float32)
        else:
            labels = None
        if apply_cmvn:
            inputs = (inputs - cmvn["mean_inputs"]) / cmvn["stddev_inputs"]
            write_binary_file(inputs, inputs_outdir)
            if labels is not None:
                labels = (labels - cmvn["mean_labels"]) / cmvn["stddev_labels"]
                write_binary_file(labels, labels_outdir)

    config_file.close()


def read_binary_file(filename, dimension=None, dtype=np.float32):
    """Read data from matlab binary file (row, col and matrix).
    Returns:
        A numpy matrix containing data of the given binary file.
    """
    if dimension is None:
        read_buffer = open(filename, 'rb')

        rows = 0;
        cols = 0
        rows = struct.unpack('<i', read_buffer.read(4))[0]
        cols = struct.unpack('<i', read_buffer.read(4))[0]

        tmp_mat = np.frombuffer(read_buffer.read(rows * cols * 4), dtype=dtype)
        mat = np.reshape(tmp_mat, (rows, cols))

        read_buffer.close()

        return mat
    else:
        fid_lab = open(filename, 'rb')
        features = np.fromfile(fid_lab, dtype=dtype)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0, 'specified dimension %s not compatible with data' % (dimension)
        features = features[:(dimension * (features.size // dimension))]
        features = features.reshape((-1, dimension))

        return features


def write_binary_file(data, output_file_name, dtype=np.float32, with_dim=False):
    data = np.asarray(data, dtype=dtype)
    fid = open(output_file_name, 'wb')
    if with_dim:
        fid.write(struct.pack('<i', data.shape[0]))
        fid.write(struct.pack('<i', data.shape[1]))
    data.tofile(fid)
    fid.close()
