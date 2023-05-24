# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:07:33 2019
Load datasets for models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division

## PyTorch dependencies
import torch
from torchvision import transforms

## Local external libraries
from Datasets.DeepShipSegments import DeepShipSegments
from Datasets.Get_Audio_Features import Get_Audio_Features


def Prepare_DataLoaders(Network_parameters, split,input_size=224):
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    
    # Data augmentation and normalization for training
    # Just normalization and resize for test
    # Data transformations as described in:
    # http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf
    if not(Network_parameters['rotation']):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(Network_parameters['center_size']),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(Network_parameters['center_size']),
                transforms.CenterCrop(input_size),
                transforms.RandomAffine(Network_parameters['degrees']),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    

        
    if Dataset == 'DeepShip':
        #Zero pad to get input to be 48 x 48
        data_transforms = Get_Audio_Features(Network_parameters['feature'],RGB=False)
        train_dataset = DeepShipSegments(data_dir, partition='train', transform = data_transforms['train'])
        val_dataset = DeepShipSegments(data_dir, partition='val', transform = data_transforms['test'])
        test_dataset = DeepShipSegments(data_dir, partition='test', transform = data_transforms['test'])        
    else:
        raise RuntimeError('Dataset not implemented') 



    image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    # Create training and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=Network_parameters['batch_size'][x],
                                                       shuffle=True,
                                                       num_workers=Network_parameters['num_workers'],
                                                       pin_memory=Network_parameters['pin_memory'])
                                                       for x in ['train', 'val','test']}

    return dataloaders_dict
    