from typing import Tuple

import mne
from mne.io.edf.edf import RawEDF
import numpy as np


def load_data(signal_path: str) -> Tuple[np.ndarray,float]:
    '''Returns the ecg and the sampling frequency.'''
    
    data: RawEDF = mne.io.read_raw_edf(signal_path)
    
    sfreq: float = data.info['sfreq']

    # channel 0 has the ecg
    # convert to milli Volt
    ecg: np.ndarray = data.get_data()[0] * 1_000

    print(f'Raw ECG {ecg.shape} at {sfreq} Hz')
    
    return ecg, sfreq
