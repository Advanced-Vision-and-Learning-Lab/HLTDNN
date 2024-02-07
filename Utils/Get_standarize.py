#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:09:00 2024

@author: jarin.ritu
"""
import pdb
import numpy as np
from scipy.io.wavfile import read
from sklearn.preprocessing import StandardScaler

def get_standardization_minibatch(dataset, batch_size=32):
    scalers = []

    # Iterate through the dataset in mini-batches
    for idx in range(0, len(dataset), batch_size):
        batch_signals = []

        # Load and preprocess signals in the current mini-batch
        for i in range(batch_size):
            if idx + i < len(dataset):
                file_path, label = dataset.segment_lists[dataset.partition][idx + i]
                sr, signal = read(file_path, mmap=False)
                signal = signal.astype(np.float32)
                batch_signals.append(signal)

        # Compute standardization parameters for the current mini-batch
        batch_signals = np.concatenate(batch_signals, axis=0)
        scaler = StandardScaler().fit(batch_signals.reshape(-1, 1))
        scalers.append(scaler)

    # Return a function that standardizes input signals
    def standardize(signal):
        return scalers[-1].transform(signal.reshape(-1, 1)).flatten()

    return standardize

