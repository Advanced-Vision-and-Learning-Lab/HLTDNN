# -*- coding: utf-8 -*-
"""
Load datasets for models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division

## PyTorch dependencies
import torch

## Local external libraries
from Datasets.DeepShipSegments import DeepShipSegments
from Datasets.Get_Audio_Features import Get_Audio_Features


def Prepare_DataLoaders(Network_parameters, split):
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    
    #Change input to network based on models
    #If TDNN or HLTDNN, number of input features is 1
    #Else (CNN), replicate input to be 3 channels
    #If number of input channels is 3 for TDNN, RGB will be set to False
    if (Network_parameters['Model_name'] == 'TDNN' and Network_parameters['TDNN_feats'][Dataset]):
        RGB = True
    else:
        RGB = False
        
    if Dataset == 'DeepShip':
        data_transforms = Get_Audio_Features(Network_parameters['feature'],RGB=RGB)
        train_dataset = DeepShipSegments(data_dir, partition='train', transform = data_transforms['train'])
        val_dataset = DeepShipSegments(data_dir, partition='val', transform = data_transforms['test'])
        test_dataset = DeepShipSegments(data_dir, partition='test', transform = data_transforms['test'])        
    else:
        raise RuntimeError('Dataset not implemented') 


    #Create dictionary of datasets
    image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    
    # Create training and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=Network_parameters['batch_size'][x],
                                                       shuffle=True,
                                                       num_workers=Network_parameters['num_workers'],
                                                       pin_memory=Network_parameters['pin_memory'])
                                                       for x in ['train', 'val','test']}

    return dataloaders_dict
    