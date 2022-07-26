# %%

import math
from typing import List

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

from inference import dataset, experiments, raw

# %% LIMIT GPU USAGE OF TENSORFLOW

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# %% LOAD DATASET SPLITS

# choose between 1 or 2
noise_version: int = 1

# shapes: (num_examples, num_samples=512, 1)
[X_train, y_train, X_val, y_val, X_test, y_test] = dataset.get_splits(noise_version)

# %% LOAD PRETRAINED MODEL

experiment= experiments.Experiment.Multibranch_LANLD

model = experiments.load_model(experiment, noise_version)

# %% MAKE PREDICTIONS ON TEST SET

# shape: (num_examples, num_samples=512, 1)
y_pred: np.ndarray = model.predict(X_test, batch_size=32, verbose=1)

# %% VISUALIZE SINGLE PREDICTION

index = 0

plt.figure(figsize=(10,5))
plt.plot(X_test[index], label='X_test')
plt.plot(y_test[index], label='y_test')
plt.plot(y_pred[index], label='y_pred')
plt.legend()
plt.show()

# %% LOAD ONE OF OUR ECGs

signal_path: str = '/shared/Delineation Dataset/AttCHR0102rcs4_Sinius/13-20-54.EDF'

ecg, old_fs = raw.load_data(signal_path)

# %% ISOLATE A BEAT OF THE ECG

new_fs = 360.0
max_num_new_fs_samples = 512

max_num_old_fs_samples = math.floor(max_num_new_fs_samples * old_fs / new_fs)
print(f'max number of samples with old sampling frequency: {max_num_old_fs_samples}')

start_sample = 7_450
# or any other index of sample

num_old_fs_samples = max_num_old_fs_samples 
# or any other number below max_num_old_fs_samples

# shape: (num_old_fs_samples,)
test_beat = ecg[start_sample: start_sample + num_old_fs_samples]
print(f'test beat shape: {test_beat.shape}')

plt.figure(figsize=(10,5))
plt.plot(test_beat)
plt.show()

# %% RESAMPLE AND ZERO-PAD TEST BEAT

num_new_fs_samples = math.ceil(len(test_beat)*new_fs/old_fs)
assert num_new_fs_samples <= 512

normBeat: List[float] = list(reversed(test_beat)) + list(test_beat) + list(reversed(test_beat))

# resample beat
res_beat: np.ndarray = resample_poly(normBeat, new_fs, old_fs)
res_beat = res_beat[num_new_fs_samples-1:2*num_new_fs_samples-1]
assert len(res_beat) == num_new_fs_samples

# zero padding and reshape
pad_res_beat: np.ndarray = np.zeros((1, 512, 1))
pad_res_beat[0, 0:num_new_fs_samples, 0] = res_beat

# %% 

sample_y_pred: np.ndarray = model.predict(pad_res_beat, batch_size=32, verbose=1)

plt.figure(figsize=(10,5))
plt.plot(pad_res_beat[0], label='sample')
plt.plot(sample_y_pred[0], label='pred')
plt.legend()
plt.show()

# %%
