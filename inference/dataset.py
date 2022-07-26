import pickle
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split


def get_splits(noise_version: int) -> List[np.ndarray]:
    '''Returns the splits of the paper's dataset.'''

    with open(f'data/dataset_nv{noise_version}.pkl', 'rb') as input:
        [X_train, y_train, X_test, y_test] = pickle.load(input)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)

    print(f'X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}')
    print(f'y_train {y_train.shape}, y_val {y_val.shape}, y_test {y_test.shape}')

    return [X_train, y_train, X_val, y_val, X_test, y_test]
