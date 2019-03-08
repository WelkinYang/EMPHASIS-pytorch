import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv1d, highwaynet

with open('./hparams.json', 'r') as f:
    hparams = json.load(f)

class EMPHASISAcousticModel(nn.Module):
    def __init__(self, in_channels, units, bank_widths, max_pooling_width, duration_highway_layers, gru_layer, spec_hidden_size, energy_hidden_size, cap_hidden_size,
                 lf0_hidden_size, activation=[nn.ReLU(), nn.Sigmoid()]):
        super(EMPHASISAcousticModel, self).__init__()
        self.convs_bank = [nn.Conv1d(in_channels=in_channels, out_channels=units, kernel_size=k, stride=1, dilation=0 if k%2 == 1 else 2,
                                     padding=(k-1)/2 if k%2 == 1 else k-1)
                           for k in bank_widths]

        self.max_pool = nn.MaxPool1d(kernel_size=max_pooling_width, stride=1, dilation=0 if max_pooling_width%2 == 1 else 2,
                                     padding=(max_pooling_width-1)/2 if max_pooling_width%2 == 1 else max_pooling_width-1)

        self.conv_projection = nn.Conv1d(in_channels=units*len(bank_widths), out_channels=units, kernel_size=3, stride=1, padding=1)

        self.duration_highway_layers = duration_highway_layers

        self.spec_gru = nn.GRU(input_size=units, hidden_size=spec_hidden_size, num_layers=gru_layer,
                               batch_first=True, bidirectional=True)

        self.energy_gru = nn.GRU(input_size=units, hidden_size=energy_hidden_size, num_layers=gru_layer,
                                 batch_first=True, bidirectional=True)

        self.cap_gru = nn.GRU(input_size=units, hidden_size=cap_hidden_size, num_layers=gru_layer,
                                 batch_first=True, bidirectional=True)

        self.lf0_gru = nn.GRU(input_size=units, hidden_size=lf0_hidden_size, num_layers=gru_layer,
                              batch_first=True, bidirectional=True)

        self.spec_linear = nn.Linear(units, hparams['spec_units'])

        self.cap_linear = nn.Linear(units, hparams['cap_units'])

        self.lf0_linear = nn.Linear(units, hparams['lf0_units'])

        self.energy_linear = nn.Linear(units, hparams['energy_units'])

        self.uv_linear = nn.Linear(units, hparams['uv_units'])

        self.activation = activation

    def forward(self, input):
        phoneme_input = input[:, :, :hparams['phoneme_input_channels']]
        emotional_prosodic_input = input[:, :, hparams['phoneme_input_channels']: -1]
        # Convolution bank: concatenate on the last axis to stack channels from all convolutions
        phoneme_conv_ouputs = torch.cat([
            Conv1d(phoneme_input, conv, nn.BatchNorm1d(conv.out_channels), self.training, activation=self.activation[0])
            for conv in self.convs_bank], dim=-1)

        emotional_prosodic_conv_outputs = torch.cat([
            Conv1d(emotional_prosodic_input, conv, self.training, None, activation=self.activation[0])
            for conv in self.convs_bank], dim=-1)

        # Maxpooling:
        phoneme_maxpool_output = self.max_pool(phoneme_conv_ouputs)
        emotional_prosodic_maxpool_outputs = self.max_pool(emotional_prosodic_conv_outputs)

        # Projection layer:
        phoneme_proj_output = Conv1d(phoneme_maxpool_output, self.conv_projection, self.training,
                                     nn.BatchNorm1d(self.conv_projection.out_channels),
                                     activation=self.activation[0])
        emotional_prosodic_proj_output = Conv1d(emotional_prosodic_maxpool_outputs, self.conv_projection, self.training,
                                                nn.BatchNorm1d(self.conv_projection.out_channels),
                                                 activation=self.activation[0])

        highway_input = torch.cat([phoneme_proj_output, emotional_prosodic_proj_output], dim=-1)

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != self.conv_projection.out_channels * 2:
            highway_input = F.linear(highway_input,
                                     weight=torch.nn.init.normal_(torch.empty(self.conv_projection.out_channels * 2, highway_input.shape[2])))

        # HighwayNet:
        for i in range(self.duration_highway_layers):
            highway_input = highwaynet(highway_input, self.activation)
        rnn_input = highway_input

        # Bidirectional RNN
        spec_rnn_output, _ = self.spec_gru(rnn_input)
        energy_rnn_output, _ = self.energy_gru(rnn_input)
        cap_rnn_output, _ = self.cap_gru(rnn_input)
        lf0_rnn_output, _ = self.lf0_gru(rnn_input)

        spec_output, _ = self.spec_linear(spec_rnn_output)
        energy_output, _ = self.energy_linear(energy_rnn_output)
        cap_output, _ = self.cap_linear(cap_rnn_output)
        lf0_output, _ = self.lf0_linear(lf0_rnn_output)
        uv_output, _ = self.uv_linear(rnn_input)

        outputs = torch.cat([spec_output, lf0_output, uv_output, cap_output, energy_output], dim=-1)

        return outputs
