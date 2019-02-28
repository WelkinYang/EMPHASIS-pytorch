import os
import sys
import json
import struct
import logging
import numpy as np

import torch
import torch.nn.functional as F

with open('./hparams.json', 'r') as f:
    hparams = json.load(f)

logger = logging.getLogger(__name__)

def Conv1d(inputs, conv, is_training, batch_norm=None, activation=None):
    # the Conv1d of pytroch chanages the channels at the 1 dim
    # [batch_size, max_time, feature_dims] -> [batch_size, feature_dims, max_time]
    inputs = torch.transpose(inputs, 1, 2)
    conv1d_output = conv(inputs)
    if batch_norm is not None:
        batch_norm_output = batch_norm(conv1d_output)
        batch_norm_output = torch.transpose(batch_norm_output, 1 ,2)
    else:
        batch_norm_output = torch.transpose(conv1d_output, 1, 2)
    if activation is not None:
        batch_norm_output = activation(batch_norm_output)
    return F.dropout(batch_norm_output, p=hparams["dropout_rate"], training=is_training)

def highwaynet(inputs, activation, units=128):
    H = F.linear(inputs, weight=torch.nn.init.normal_(torch.empty(units, inputs.size(2))))
    H = activation[0](H)
    T = F.linear(inputs, weight=torch.nn.init.normal_(torch.empty(units, inputs.size(2))), bias=nn.init.constant_(torch.empty(1, 1, units), -0.1))
    T = activation[1](T)
    return H * T + inputs * (1.0 - T)

def calculate_cmvn(name, config_dir, output_dir):
    """Calculate mean and var."""
    logger.info("Calculating mean and var of %s" % name)
    config_filename = open(os.path.join(config_dir, name + '.lst'))

    inputs_frame_count, labels_frame_count = 0, 0
    for line in config_filename:
        utt_id, inputs_path, labels_path = line.strip().split()
        logger.info("Reading utterance %s" % utt_id)
        inputs = read_binary_file(inputs_path, hparams['in_channels'])
        labels = read_binary_file(labels_path, hparams['target_channels'])
        if inputs_frame_count == 0:    # create numpy array for accumulating
            ex_inputs = np.sum(inputs, axis=0)
            ex2_inputs = np.sum(inputs**2, axis=0)
            ex_labels = np.sum(labels, axis=0)
            ex2_labels = np.sum(labels**2, axis=0)
        else:
            ex_inputs += np.sum(inputs, axis=0)
            ex2_inputs += np.sum(inputs**2, axis=0)
            ex_labels += np.sum(labels, axis=0)
            ex2_labels += np.sum(labels**2, axis=0)
        inputs_frame_count += len(inputs)
        labels_frame_count += len(labels)

    mean_inputs = ex_inputs / inputs_frame_count
    stddev_inputs = np.sqrt(ex2_inputs / inputs_frame_count - mean_inputs**2)
    stddev_inputs[stddev_inputs < 1e-20] = 1e-20

    mean_labels = ex_labels / labels_frame_count
    stddev_labels = np.sqrt(ex2_labels / labels_frame_count - mean_labels**2)
    stddev_labels[stddev_labels < 1e-20] = 1e-20

    cmvn_name = os.path.join(output_dir, name + "_cmvn.npz")
    np.savez(cmvn_name,
             mean_inputs=mean_inputs,
             stddev_inputs=stddev_inputs,
             mean_labels=mean_labels,
             stddev_labels=stddev_labels)
    config_filename.close()
    logger.info("Wrote to %s" % cmvn_name)


def convert_to(name, config_dir, output_dir, apply_cmvn=True):
    os.mkdir(output_dir)
    cmvn = np.load(os.path.join(output_dir, "train_cmvn.npz"))
    config_file = open(os.path.join(config_dir, name + ".lst"))
    for line in config_file:
        if name != 'test':
            utt_id, inputs_path, labels_path = line.strip().split()
            inputs_outdir = os.path.join(output_dir, name) + f'{utt_id}.lab'
            labels_outdir = os.path.join(output_dir, name) + f'{utt_id}.cmp'
        else:
            utt_id, inputs_path = line.strip().split()
            inputs_outdir = os.path.join(output_dir, name) + f'{utt_id}.lab'

        logger.info(f'Writing utterance {utt_id} ...')
        inputs = read_binary_file(inputs_path, hparams['in_channels']).astype(np.float64)
        if name != 'test':
            labels = read_binary_file(labels_path, hparams['target_channels']).astype(np.float64)
        else:
            labels = None
        if apply_cmvn:
            inputs = (inputs - cmvn["mean_inputs"]) / cmvn["stddev_inputs"]
            write_binary_file(inputs, inputs_outdir)
            if labels is not None:
                labels = (labels - cmvn["mean_labels"]) / cmvn["stddev_labels"]
                write_binary_file(labels, labels_outdir)

    config_file.close()

def read_binary_file(filename, dimension=None):
    """Read data from matlab binary file (row, col and matrix).
    Returns:
        A numpy matrix containing data of the given binary file.
    """
    if dimension is None:
        read_buffer = open(filename, 'rb')

        rows = 0; cols= 0
        rows = struct.unpack('<i', read_buffer.read(4))[0]
        cols = struct.unpack('<i', read_buffer.read(4))[0]

        tmp_mat = np.frombuffer(read_buffer.read(rows * cols * 4), dtype=np.float32)
        mat = np.reshape(tmp_mat, (rows, cols))

        read_buffer.close()

        return mat
    else:
        fid_lab = open(filename, 'rb')
        features = np.fromfile(fid_lab, dtype=np.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
        features = features[:(dimension * (features.size // dimension))]
        features = features.reshape((-1, dimension))

        return features


def write_binary_file(data, output_file_name, with_dim=False):
    data = np.asarray(data, np.float32)
    fid = open(output_file_name, 'wb')
    if with_dim:
        fid.write(struct.pack('<i', data.shape[0]))
        fid.write(struct.pack('<i', data.shape[1]))
    data.tofile(fid)
    fid.close()