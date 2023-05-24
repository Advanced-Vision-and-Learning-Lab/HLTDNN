#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:26:34 2023
Select features for audio extraction
@author: jarin.ritu
"""
from torchvision import transforms
import torchaudio.transforms as T
from torchaudio.transforms import FrequencyMasking, TimeMasking
from nnAudio import features


def Get_Audio_Features(feature, RGB=False):
    
    if RGB:
        num_channels = 3
    else:
        num_channels = 1
        
    sample_rate = 16000
    
    #Based on desired feature, return transformation
    
    if feature == 'Mel_Spectrogram': #(Finished)
        #Return Mel Spectrogram that is 3 x 48 x 48
        
        signal_transform = T.MelSpectrogram(sample_rate,n_mels=40,win_length=int(.25*sample_rate),
                                            hop_length=int(.064*sample_rate),
                                            n_fft=int(.25*sample_rate))
        
        
        train_transforms = transforms.Compose([
            signal_transform,
            FrequencyMasking(freq_mask_param=48),
            TimeMasking(time_mask_param=192),
            transforms.Pad((1,4,0,4)),
            transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
            # #Using pretrained model, need to use ImageNet values
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        
        #Zero pad to get input to be 48 x 48
        test_transforms = transforms.Compose([
                signal_transform,
                transforms.Pad((1,4,0,4)),
                transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
                # #Using pretrained model, need to use ImageNet values
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        data_transforms = { 'train':train_transforms,
                          'test': test_transforms
            }
        
       
    elif feature == 'MFCC':#(Finished)
        signal_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=16, 
                                        melkwargs={"n_fft": int(.25*sample_rate), "win_length": int(.25*sample_rate), 
                                                   "hop_length": int(.064*sample_rate),
                                                   "n_mels": 48, "center": False})

        
        
        train_transforms = transforms.Compose([
            signal_transform,
            FrequencyMasking(freq_mask_param=48),
            TimeMasking(time_mask_param=192),
            transforms.Pad((1,0,4,0)),
            transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
            # #Using pretrained model, need to use ImageNet values
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        
        #Zero pad to get input to be 48 x 48
        test_transforms = transforms.Compose([
                signal_transform,
                transforms.Pad((1,0,4,0)),
                transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
                # #Using pretrained model, need to use ImageNet values
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        data_transforms = { 'train':train_transforms,
                          'test': test_transforms
            }

    elif feature == 'STFT': #(Finished)
        #pdb.set_trace()
        
        signal_transform = features.STFT(sr=sample_rate,n_fft=int(.25*sample_rate), 
                                         hop_length=int(.064*sample_rate),
                                         win_length=int(.25*sample_rate), 
                                         output_format='Magnitude',
                                         freq_bins=48,verbose=False)
            
        
        train_transforms = transforms.Compose([
            signal_transform,
            FrequencyMasking(freq_mask_param=48),
            TimeMasking(time_mask_param=192),
            transforms.Pad((1,0,0,0)),
            transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
            # #Using pretrained model, need to use ImageNet values
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        
        #Zero pad to get input to be 48 x 48
        test_transforms = transforms.Compose([
                signal_transform,
                transforms.Pad((1,0,0,0)),
                transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
                # #Using pretrained model, need to use ImageNet values
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        data_transforms = { 'train':train_transforms,
                          'test': test_transforms
            }
        
    elif feature == 'GFCC': #(Finished)
      
        signal_transform = features.Gammatonegram(sr=sample_rate,hop_length=int(.064*sample_rate),n_fft=int(.25*sample_rate),
                                                  verbose=False,n_bins=64)
        
        train_transforms = transforms.Compose([
            signal_transform,
            FrequencyMasking(freq_mask_param=48),
            TimeMasking(time_mask_param=192),
            transforms.Pad((1,0,0,0)),
            transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
            # #Using pretrained model, need to use ImageNet values
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        
        #Zero pad to get input to be 48 x 48
        test_transforms = transforms.Compose([
                signal_transform,
                transforms.Pad((1,0,0,0)),
                transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
                # #Using pretrained model, need to use ImageNet values
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        data_transforms = { 'train':train_transforms,
                          'test': test_transforms
            }

    elif feature == 'CQT': #(Finished)

        signal_transform = features.CQT(sr=sample_rate, n_bins=64, hop_length=int(.064*sample_rate),
                                        verbose=False)


        
        train_transforms = transforms.Compose([
            signal_transform,
            FrequencyMasking(freq_mask_param=48),
            TimeMasking(time_mask_param=192),
            transforms.Pad((1,0,0,0)),
            transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
            # #Using pretrained model, need to use ImageNet values
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        
        #Zero pad to get input to be 48 x 48
        test_transforms = transforms.Compose([
                signal_transform,
                transforms.Pad((1,0,0,0)),
                transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
                # #Using pretrained model, need to use ImageNet values
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        data_transforms = { 'train':train_transforms,
                          'test': test_transforms
            }
        
    elif feature == 'VQT': #(Finished)
        signal_transform = features.VQT(sr=sample_rate,hop_length=int(.064*sample_rate),
                                        n_bins=64,earlydownsample=False,verbose=False)

        train_transforms = transforms.Compose([
            signal_transform,
            FrequencyMasking(freq_mask_param=48),
            TimeMasking(time_mask_param=192),
            transforms.Pad((1,0,0,0)),
            transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
            # #Using pretrained model, need to use ImageNet values
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        
        #Zero pad to get input to be 48 x 48
        test_transforms = transforms.Compose([
                signal_transform,
                transforms.Pad((1,0,0,0)),
                transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
                # #Using pretrained model, need to use ImageNet values
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        data_transforms = { 'train':train_transforms,
                          'test': test_transforms
            }
       
    else:
        raise RuntimeError('{} not implemented'.format(feature))
            
    
    return data_transforms