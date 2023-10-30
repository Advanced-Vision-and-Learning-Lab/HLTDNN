#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:26:34 2023
Generate audio feature transforms
@author: jarin.ritu
"""
from torchvision import transforms
import torchaudio.transforms as T
from nnAudio import features
import librosa
import numpy as np
import pdb
import torch
import torch.nn as nn
from nnAudio import features


def vqt_lib(x):
    x_numpy = x.numpy().astype(np.float32)
    signal = librosa.vqt(x_numpy, sr=16000,hop_length=int((64/1000)*16000),
                                        n_bins=64)
    signal = torch.from_numpy(signal)
    return signal

def Get_Audio_Features(input_features, data, sample_rate=16000, window_length=250, 
                       hop_length=64, RGB=False, pretrained=False, device="cpu"):
    
    #Convert window and hop length to ms
    window_length /= 1000
    hop_length /= 1000
    
    if RGB:
        num_channels = 3
    else:
        num_channels = 1
    
    #Based on desired feature, return transformation
    Mel_Spectrogram = nn.Sequential(features.mel.MelSpectrogram(sample_rate,n_mels=40,win_length=int(window_length*sample_rate),
                                            hop_length=int(hop_length*sample_rate),
                                            n_fft=int(window_length*sample_rate), verbose=False), nn.ZeroPad2d((1,4,0,4)))
        
    
        #Return MFCC that is 16 x 48
    MFCC = nn.Sequential(features.mel.MFCC(sr=sample_rate, n_mfcc=16, 
                                        n_fft=int(window_length*sample_rate), 
                                                win_length=int(window_length*sample_rate), 
                                                hop_length=int(hop_length*sample_rate),
                                                n_mels=48, center=False, verbose=False), nn.ZeroPad2d((1,0,0,0)))

        #Return STFT that is 48 x 48
    STFT = nn.Sequential(features.STFT(sr=sample_rate,n_fft=int(window_length*sample_rate), 
                                        hop_length=int(hop_length*sample_rate),
                                        win_length=int(window_length*sample_rate), 
                                        output_format='Magnitude',
                                        freq_bins=48,verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        #Return GFCC that is 64 x 48
    GFCC = nn.Sequential(features.Gammatonegram(sr=sample_rate,
                                                hop_length=int(hop_length*sample_rate),
                                                n_fft=int(window_length*sample_rate),
                                                verbose=False,n_bins=64), nn.ZeroPad2d((1,0,0,0)))
        

        #Return CQT that is 64 x 48
    CQT = nn.Sequential(features.CQT(sr=sample_rate, n_bins=64, 
                                        hop_length=int(hop_length*sample_rate),
                                        verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        #Return VQT that is 64 x 48
    VQT = nn.Sequential(features.VQT(sr=sample_rate,hop_length=int(hop_length*sample_rate),
                                        n_bins=64,earlydownsample=False,verbose=False), nn.ZeroPad2d((1,0,0,0)))

    feature_extract = {'Mel_Spectrogram': Mel_Spectrogram, 'MFCC': MFCC, 'STFT': STFT, 'GFCC': GFCC, 'CQT': CQT, 'VQT': VQT}
            
    transformed_features = []
    
    for feature in input_features:
        transformed_features.append(feature_extract[feature](data))
    combined_features = torch.stack(transformed_features, dim=1)
    
    return combined_features
