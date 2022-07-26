from enum import Enum
from typing import Dict, Callable, Union, Any

import numpy as np
import keras
from keras import losses
from keras import backend as K
from keras.models import Sequential, Model

import deepFilter.dl_models as models


class Experiment(Enum):
    '''Enumerator class for the different experiments.'''

    FCN_DAE = 'FCN_DAE'
    DRNN = 'DRNN'
    Vanilla_L = 'Vanilla_L'
    Vanilla_NL = 'Vanilla_NL'
    Multibranch_LANL = 'Multibranch_LANL'
    Multibranch_LANLD = 'Multibranch_LANLD'


def ssd_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-2)


def combined_ssd_mad_loss(y_true, y_pred):
    return K.max(K.square(y_true - y_pred), axis=-2) * 50 + K.sum(K.square(y_true - y_pred), axis=-2)


def mad_loss(y_true, y_pred):
    return K.max(K.square(y_pred - y_true), axis=-2)


SIGNAL_SIZE: int = 512


experiment_to_model: Dict[Experiment, Union[Model,Sequential]] = {
    Experiment.FCN_DAE: models.FCN_DAE(signal_size=SIGNAL_SIZE),
    Experiment.DRNN: models.DRRN_denoising(signal_size=SIGNAL_SIZE),
    Experiment.Vanilla_L: models.deep_filter_vanilla_linear(signal_size=SIGNAL_SIZE),
    Experiment.Vanilla_NL: models.deep_filter_vanilla_Nlinear(signal_size=SIGNAL_SIZE),
    Experiment.Multibranch_LANL: models.deep_filter_I_LANL(signal_size=SIGNAL_SIZE),
    Experiment.Multibranch_LANLD: models.deep_filter_model_I_LANL_dilated(signal_size=SIGNAL_SIZE)
}


experiment_to_loss: Dict[Experiment, Union[str, Callable[[np.ndarray,np.ndarray], Any]]] = {
    Experiment.DRNN: 'mse',
    Experiment.FCN_DAE: ssd_loss,
    Experiment.Vanilla_L: combined_ssd_mad_loss,
    Experiment.Vanilla_NL: combined_ssd_mad_loss,
    Experiment.Multibranch_LANL: combined_ssd_mad_loss,
    Experiment.Multibranch_LANLD: combined_ssd_mad_loss
}


def load_model(experiment: Experiment, noise_version: int) -> Union[Model,Sequential]:
    '''Returns a pretrained model.'''

    model = experiment_to_model[experiment]

    model.compile(loss=experiment_to_loss[experiment],
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=[losses.mean_squared_error, losses.mean_absolute_error, ssd_loss, mad_loss])

    model.load_weights(f'./noise_version_{noise_version}/{experiment.value}_weights.best.hdf5')

    print(f'{experiment.value} loaded')

    return model
