#0 -*- coding: utf-8 -*-
"""
Main script for baseline and histogram layer(s) networks experiments
demo.py
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import argparse
import random

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Utils.Network_functions import initialize_model, train_model, test_model
from Utils.RBFHistogramPooling import HistogramLayer
from Utils.Save_Results import save_results
from Utils.Get_Optimizer import get_optimizer
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders

def main(Params):
    
    # Name of dataset
    Dataset = Params['Dataset']
    
    # Model(s) to be used
    model_name = Params['Model_name']
    
    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]
    
    # Number of runs and/or splits for dataset
    numRuns = Params['Splits'][Dataset]
    
    # Number of bins and input convolution feature maps after channel-wise pooling
    numBins = Params['numBins']
    num_feature_maps = Params['out_channels'][model_name]
    
    # Local area of feature map after histogram layer
    feat_map_size = Params['feat_map_size']
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using", torch.cuda.device_count(), "GPUs!")
    
    print('Starting Experiments...')
    for split in range(0, numRuns):
        
        #Set random state for reproducibility
        torch.manual_seed(split)
        np.random.seed(split)
        random.seed(split)
        torch.cuda.manual_seed(split)
        torch.cuda.manual_seed_all(split)
        
        # Keep track of the bins and widths as these values are updated each
        # epoch
        saved_bins = np.zeros((Params['num_epochs'] + 1,
                               numBins * int(num_feature_maps / (feat_map_size * numBins))))
        saved_widths = np.zeros((Params['num_epochs'] + 1,
                                 numBins * int(num_feature_maps / (feat_map_size * numBins))))

        histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
                                         Params['kernel_size'][model_name],
                                         num_bins=numBins, stride=Params['stride'],
                                         normalize_count=Params['normalize_count'],
                                         normalize_bins=Params['normalize_bins'])


        # Initialize the histogram model for this run
        model_ft, input_size, feature_extraction_layer = initialize_model(model_name, num_classes,
                                                Params['in_channels'][model_name],
                                                # len(Params['feature']),
                                                num_feature_maps,
                                                feature_extract=Params['feature_extraction'],
                                                histogram=Params['histogram'],
                                                histogram_layer=histogram_layer,
                                                parallel=Params['parallel'],
                                                use_pretrained=Params['use_pretrained'],
                                                add_bn=Params['add_bn'],
                                                scale=Params['scale'],
                                                feat_map_size=feat_map_size,
                                                TDNN_feats=(Params['TDNN_feats'][Dataset] * len(Params['feature'])),
                                                input_features = Params['feature'])

        # Send the model to GPU if available, use multiple if available
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_ft = nn.DataParallel(model_ft)
        model_ft = model_ft.to(device)
        feature_extraction_layer = feature_extraction_layer.to(device)
        # Print number of trainable parameters
        num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
        print("Initializing Datasets and Dataloaders...")

        # Create training and validation dataloaders
        dataloaders_dict = Prepare_DataLoaders(Params)

        # Save the initial values for bins and widths of histogram layer
        # Set optimizer for model
        if (Params['histogram']):
            reduced_dim = int((num_feature_maps / feat_map_size) / (numBins))
            if (Params['in_channels'][model_name] == reduced_dim):
                dim_reduced = False
                saved_bins[0, :] = model_ft.module.histogram_layer.centers.detach().cpu().numpy()
                saved_widths[0, :] = model_ft.module.histogram_layer.widths.reshape(
                    -1).detach().cpu().numpy()
            else:
                dim_reduced = True
                saved_bins[0, :] = model_ft.module.histogram_layer[
                    -1].centers.detach().cpu().numpy()
                saved_widths[0, :] = model_ft.module.histogram_layer[-1].widths.reshape(
                    -1).detach().cpu().numpy()
        else:
            saved_bins = None
            saved_widths = None
            dim_reduced = None
        # Setup the loss fxn and scheduler
        criterion = nn.CrossEntropyLoss()
        scheduler = None
        
        # Define optimizer for updating weights
        params = model_ft.parameters()
        optimizer_ft = get_optimizer(params, Params['optimizer'], lr=Params['lr'])
        # Train and evaluate
        train_dict = train_model(model_ft, dataloaders_dict, criterion, 
                                 optimizer_ft, 
                                 device,
                                 feature_extraction_layer,
                                 saved_bins=saved_bins, saved_widths=saved_widths,
                                 histogram=Params['histogram'],
                                 num_epochs=Params['num_epochs'],
                                 scheduler=scheduler,
                                 dim_reduced=dim_reduced)
        test_dict = test_model(dataloaders_dict['test'], model_ft, feature_extraction_layer,criterion,
                               device)

        # Save results
        if (Params['save_results']):
            save_results(train_dict, test_dict, split, Params, num_params)
            del train_dict, test_dict
            torch.cuda.empty_cache()

        if (Params['histogram']):
            print('**********Run ' + str(split + 1) + ' For '
                  + Params['hist_model'] + ' Finished**********')
        else:
            print('**********Run ' + str(split + 1) + ' For GAP_' +
                  model_name + ' Finished**********')

def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments (default: True)')
    parser.add_argument('--folder', type=str, default='Saved_Models/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='TDNN',
                        help='Select baseline model architecture')
    parser.add_argument('--histogram', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use histogram model or baseline global average pooling (GAP), --no-histogram (GAP) or --histogram')
    parser.add_argument('--data_selection', type=int, default=0,
                        help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
    parser.add_argument('--audio_feature', nargs='+', default=['GFCC'],
                        help='Audio feature for extraction')
    parser.add_argument('--optimizer', type = str, default = 'Adagrad',
                       help = 'Select optimizer')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    params = Parameters(args)
    main(params)
