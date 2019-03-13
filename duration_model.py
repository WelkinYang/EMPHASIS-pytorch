import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv1d, highwaynet

with open('./hparams.json', 'r') as f:
    hparams = json.load(f)

class EMPHASISDurationModel(nn.Module):
    def __init__(self, in_channels, units, bank_widths, max_pooling_width, highway_layers, gru_layer, duration_hidden_size, activation=[nn.ReLU(), nn.Sigmoid()]):
        super(EMPHASISDurationModel, self).__init__()
        self.bank_widths = bank_widths

        self.phoneme_convs_bank = [
            nn.Conv1d(in_channels=hparams['phoneme_in_channels'], out_channels=units, kernel_size=k).cuda()
            for k in bank_widths]

        self.emotional_prosodic_convs_bank = [
            nn.Conv1d(in_channels=hparams['emotional_prosodic_in_channels'], out_channels=units, kernel_size=k).cuda()
            for k in bank_widths]

        self.max_pool_width = max_pooling_width

        self.max_pool = nn.MaxPool1d(kernel_size=max_pooling_width, stride=1)

        self.conv_projection = nn.Conv1d(in_channels=units*len(bank_widths), out_channels=units, kernel_size=3, stride=1, padding=1)

        self.highway_layers = highway_layers

        self.gru = nn.GRU(input_size=units, hidden_size=duration_hidden_size, num_layers=gru_layer, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(duration_hidden_size*2, 1)

        self.activation = activation

    def forward(self, input):
        # Convolution bank: concatenate on the last axis to stack channels from all convolutions
        phoneme_input = input[:, :, :hparams['phoneme_input_channels']]
        emotional_prosodic_input = input[:, :, hparams['phoneme_input_channels']: -1]
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
        if highway_input.shape[2] != 128:
            highway_input = F.linear(highway_input,
                                     weight=torch.nn.init.normal_(torch.empty(128, highway_input.shape[2])))

        # HighwayNet:
        for i in range(self.highway_layers):
            highway_input = highwaynet(highway_input, self.activation)
        rnn_input = highway_input

        # Bidirectional RNN
        outputs, _ = self.gru(rnn_input)

        # Outputs [batch_size, phoneme_num, hidden_size * directions] -> [batch_size, phoneme_num]
        # the value is frame nums of the phoneme
        outputs = self.linear(outputs).squeeze()

        return outputs

