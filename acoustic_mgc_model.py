import json

import torch
import torch.nn as nn
from utils import Conv1d, MaxPool1d, HighwayNet

with open('./hparams.json', 'r') as f:
    hparams = json.load(f)


class EMPHASISAcousticMgcModel(nn.Module):
    def __init__(self, units, bank_widths, max_pooling_width, duration_highway_layers, gru_layer,
                 mgc_hidden_size, bap_hidden_size,
                 lf0_hidden_size, activation=[nn.ReLU(), nn.Sigmoid()]):
        super(EMPHASISAcousticMgcModel, self).__init__()

        self.bank_widths = bank_widths

        self.phoneme_convs_bank = nn.ModuleList([
            nn.Conv1d(in_channels=hparams['phoneme_in_channels'], out_channels=units, kernel_size=k)
            for k in bank_widths])

        self.emotional_prosodic_convs_bank = nn.ModuleList([
            nn.Conv1d(in_channels=hparams['emotional_prosodic_in_channels'], out_channels=units, kernel_size=k)
            for k in bank_widths])

        self.max_pool_width = max_pooling_width

        self.max_pool = nn.MaxPool1d(kernel_size=max_pooling_width, stride=1)

        self.conv_projection = nn.Conv1d(in_channels=units * len(bank_widths), out_channels=units, kernel_size=3,
                                         stride=1, padding=1)

        self.highway_net = HighwayNet(activation=activation)

        self.duration_highway_layers = duration_highway_layers

        self.batch_norm = nn.BatchNorm1d(self.conv_projection.out_channels)

        self.highway_linear = nn.Linear(self.conv_projection.out_channels * 2, 128)

        self.mgc_gru = nn.GRU(input_size=units, hidden_size=(mgc_hidden_size + bap_hidden_size + lf0_hidden_size),
                              num_layers=gru_layer,
                              batch_first=True, bidirectional=True)

        # self.bap_gru = nn.GRU(input_size=units, hidden_size=bap_hidden_size, num_layers=gru_layer,
        # batch_first=True, bidirectional=True)

        # self.lf0_gru = nn.GRU(input_size=units, hidden_size=lf0_hidden_size, num_layers=gru_layer,
        # batch_first=True, bidirectional=True)

        self.mgc_linear = nn.Linear((mgc_hidden_size + bap_hidden_size + lf0_hidden_size) * 2,
                                    hparams['mgc_units'] + hparams['bap_units'] + hparams['lf0_units'])

        # self.bap_linear = nn.Linear(bap_hidden_size * 2, hparams['bap_units'])

        # self.lf0_linear = nn.Linear(lf0_hidden_size * 2, hparams['lf0_units'])

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
                                     self.batch_norm,
                                     activation=self.activation[0])
        emotional_prosodic_proj_output = Conv1d(emotional_prosodic_maxpool_outputs, self.conv_projection, self.training,
                                                self.batch_norm,
                                                activation=self.activation[0])

        highway_input = torch.cat([phoneme_proj_output, emotional_prosodic_proj_output], dim=-1)

        # Handle dimensionality mismatch
        if highway_input.shape[2] != 128:
            highway_input = self.highway_linear(highway_input)

        # HighwayNet:
        for i in range(self.duration_highway_layers):
            highway_input = self.highway_net(highway_input)
        rnn_input = highway_input

        # Bidirectional RNN
        # Flatten parameters
        self.mgc_gru.flatten_parameters()
        # self.bap_gru.flatten_parameters()
        # self.lf0_gru.flatten_parameters()

        mgc_rnn_output, _ = self.mgc_gru(rnn_input)
        # bap_rnn_output, _ = self.bap_gru(rnn_input)
        # lf0_rnn_output, _ = self.lf0_gru(rnn_input)

        mgc_output = self.mgc_linear(mgc_rnn_output)
        # bap_output = self.bap_linear(bap_rnn_output)
        # lf0_output = self.lf0_linear(lf0_rnn_output)
        uv_output = self.uv_linear(rnn_input)

        # outputs = torch.cat([mgc_output, bap_output, lf0_output], dim=-1), uv_output

        outputs = mgc_output, uv_output

        return outputs
