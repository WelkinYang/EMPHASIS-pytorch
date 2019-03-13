import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv1d, MaxPool1d, highwaynet

with open('./hparams.json', 'r') as f:
    hparams = json.load(f)


class EMPHASISAcousticModel(nn.Module):
    def __init__(self, in_channels, units, bank_widths, max_pooling_width, duration_highway_layers, gru_layer,
                 spec_hidden_size, energy_hidden_size, cap_hidden_size,
                 lf0_hidden_size, activation=[nn.ReLU(), nn.Sigmoid()]):
        super(EMPHASISAcousticModel, self).__init__()

        self.bank_widths = bank_widths

        self.phoneme_convs_bank = [
            nn.Conv1d(in_channels=hparams['phoneme_in_channels'], out_channels=units, kernel_size=k).cuda()
            for k in bank_widths]

        self.emotional_prosodic_convs_bank = [
            nn.Conv1d(in_channels=hparams['emotional_prosodic_in_channels'], out_channels=units, kernel_size=k).cuda()
            for k in bank_widths]

        self.max_pool_width = max_pooling_width

        self.max_pool = nn.MaxPool1d(kernel_size=max_pooling_width, stride=1)

        self.conv_projection = nn.Conv1d(in_channels=units * len(bank_widths), out_channels=units, kernel_size=3,
                                         stride=1, padding=1)

        self.duration_highway_layers = duration_highway_layers

        self.spec_gru = nn.GRU(input_size=units, hidden_size=spec_hidden_size, num_layers=gru_layer,
                               batch_first=True, bidirectional=True)

        self.energy_gru = nn.GRU(input_size=units, hidden_size=energy_hidden_size, num_layers=gru_layer,
                                 batch_first=True, bidirectional=True)

        self.cap_gru = nn.GRU(input_size=units, hidden_size=cap_hidden_size, num_layers=gru_layer,
                              batch_first=True, bidirectional=True)

        self.lf0_gru = nn.GRU(input_size=units, hidden_size=lf0_hidden_size, num_layers=gru_layer,
                              batch_first=True, bidirectional=True)

        self.spec_linear = nn.Linear(spec_hidden_size * 2, hparams['spec_units'])

        self.cap_linear = nn.Linear(cap_hidden_size * 2, hparams['cap_units'])

        self.lf0_linear = nn.Linear(lf0_hidden_size * 2, hparams['lf0_units'])

        self.energy_linear = nn.Linear(energy_hidden_size * 2, hparams['energy_units'])

        self.uv_linear = nn.Linear(units, hparams['uv_units'])

        self.activation = activation

    def forward(self, input):
        phoneme_input = input[:, :, :hparams['phoneme_in_channels']]
        emotional_prosodic_input = input[:, :, hparams['phoneme_in_channels']:]
        # Convolution bank: concatenate on the last axis to stack channels from all convolutions
        phoneme_conv_outputs = torch.cat([
            Conv1d(phoneme_input, conv, self.training, None, activation=self.activation[0],
                   padding=self.bank_widths[i] - 1)
            for i, conv in enumerate(self.phoneme_convs_bank)], dim=-1)

        emotional_prosodic_conv_outputs = torch.cat([
            Conv1d(emotional_prosodic_input, conv, self.training, None, activation=self.activation[0],
                   padding=self.bank_widths[i] - 1)
            for i, conv in enumerate(self.emotional_prosodic_convs_bank)], dim=-1)

        # Maxpooling:
        phoneme_maxpool_output = MaxPool1d(phoneme_conv_outputs, self.max_pool, self.max_pool_width - 1)
        emotional_prosodic_maxpool_outputs = MaxPool1d(emotional_prosodic_conv_outputs, self.max_pool,
                                                       self.max_pool_width - 1)

        # Projection layer:
        phoneme_proj_output = Conv1d(phoneme_maxpool_output, self.conv_projection, self.training,
                                     nn.BatchNorm1d(self.conv_projection.out_channels).cuda(),
                                     activation=self.activation[0])
        emotional_prosodic_proj_output = Conv1d(emotional_prosodic_maxpool_outputs, self.conv_projection, self.training,
                                                nn.BatchNorm1d(self.conv_projection.out_channels).cuda(),
                                                activation=self.activation[0])

        highway_input = torch.cat([phoneme_proj_output, emotional_prosodic_proj_output], dim=-1)

        # Handle dimensionality mismatch
        if highway_input.shape[2] != 128:
            highway_input = F.linear(highway_input,
                                     weight=torch.nn.init.normal_(
                                         torch.empty(128, self.conv_projection.out_channels * 2)).cuda())

        # HighwayNet:
        for i in range(self.duration_highway_layers):
            highway_input = highwaynet(highway_input, self.activation)
        rnn_input = highway_input

        # Bidirectional RNN
        spec_rnn_output, _ = self.spec_gru(rnn_input)
        energy_rnn_output, _ = self.energy_gru(rnn_input)
        cap_rnn_output, _ = self.cap_gru(rnn_input)
        lf0_rnn_output, _ = self.lf0_gru(rnn_input)

        spec_output = self.spec_linear(spec_rnn_output)
        energy_output = self.energy_linear(energy_rnn_output)
        cap_output = self.cap_linear(cap_rnn_output)
        lf0_output = self.lf0_linear(lf0_rnn_output)
        uv_output = self.uv_linear(rnn_input)

        outputs = torch.cat([spec_output, lf0_output, cap_output, energy_output], dim=-1), uv_output

        return outputs
