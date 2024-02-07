#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:36:21 2024

@author: jarin.ritu
"""

import pdb
import numpy as np
from scipy.io.wavfile import read
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def get_min_max_minibatch_zero(dataset, batch_size=32):
    min_values = []
    max_values = []

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

        # Compute min and max values for the current mini-batch
        batch_signals = np.concatenate(batch_signals, axis=0)
        current_min = np.min(batch_signals)
        current_max = np.max(batch_signals)

        min_values.append(current_min)
        max_values.append(current_max)


    # Calculate overall min and max values
    overall_min = np.min(min_values)
    overall_max = np.max(max_values)


    # Define normalization function
    def normalize(signal):
        return (signal - overall_min) / (overall_max - overall_min)

    return normalize
