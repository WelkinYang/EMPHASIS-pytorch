from acoustic_model import EMPHASISAcousticModel
from duration_model import EMPHASISDurationModel
from acoustic_mgc_model import EMPHASISAcousticMgcModel
from acoustic_dcbhg_model import EMPHASISAcousticDcbhgMgcModel

import json
import logging
with open('./hparams.json', 'r') as f:
    hparams = json.load(f)

logger = logging.getLogger(__name__)

def create_train_model(model_type):
    if model_type == "acoustic":
        model = build_acoustic_model()
    elif model_type == "acoustic_mgc":
        model = build_acoustic_mgc_model()
    elif model_type == 'acoustic_dcbhg_mgc':
        model = build_acoustic_dcbhg_mgc_model()
    elif model_type == 'duration':
        model = build_duration_model()
    else:
        logger.error(f'this {model_type} is not supported!')
    model.train()
    return model

def build_duration_model():
    model = EMPHASISDurationModel(
        in_channels=hparams['in_channels'],
        units=hparams['duration_units'],
        bank_widths=hparams['duration_convolutional_bank_widths'],
        max_pooling_width=hparams['duration_max_pooling_width'],
        highway_layers=hparams['duration_highway_layers'],
        gru_layer=hparams['duration_gru_layer'],
        duration_hidden_size=hparams['duration_hidden_size']
    )
    return model

def build_acoustic_model():
    model = EMPHASISAcousticModel(
        in_channels=hparams['in_channels'],
        units=hparams['acoustic_units'],
        bank_widths=hparams['acoustic_convolutional_bank_widths'],
        max_pooling_width=hparams['acoustic_max_pooling_width'],
        duration_highway_layers=hparams['acoustic_highway_layers'],
        gru_layer=hparams['acoustic_gru_layer'],
        spec_hidden_size=hparams['spec_hidden_size'],
        energy_hidden_size=hparams['energy_hidden_size'],
        cap_hidden_size=hparams['cap_hidden_size'],
        lf0_hidden_size=hparams['lf0_hidden_size']
    )
    return model

def build_acoustic_mgc_model():
    model = EMPHASISAcousticMgcModel(
        in_channels=hparams['in_channels'],
        units=hparams['acoustic_units'],
        bank_widths=hparams['acoustic_convolutional_bank_widths'],
        max_pooling_width=hparams['acoustic_max_pooling_width'],
        duration_highway_layers=hparams['acoustic_highway_layers'],
        gru_layer=hparams['acoustic_gru_layer'],
        mgc_hidden_size=hparams['mgc_hidden_size'],
        bap_hidden_size=hparams['bap_hidden_size'],
        lf0_hidden_size=hparams['lf0_hidden_size']
    )
    return model

def build_acoustic_dcbhg_mgc_model():
    model = EMPHASISAcousticDcbhgMgcModel(
        in_channels=hparams['in_channels'],
        units=hparams['acoustic_units'],
        bank_widths=hparams['acoustic_convolutional_bank_widths'],
        max_pooling_width=hparams['acoustic_max_pooling_width'],
        duration_highway_layers=hparams['acoustic_highway_layers'],
        gru_layer=hparams['acoustic_gru_layer'],
        mgc_hidden_size=hparams['mgc_hidden_size'],
        bap_hidden_size=hparams['bap_hidden_size'],
        lf0_hidden_size=hparams['lf0_hidden_size']
    )
    return model
