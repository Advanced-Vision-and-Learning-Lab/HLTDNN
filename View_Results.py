# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019
Generate results from saved models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import os
from sklearn.metrics import matthews_corrcoef
import pickle
import argparse

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Utils.Generate_TSNE_visual import Generate_TSNE_visual
from Utils.Class_information import Class_names
from Demo_Parameters import Parameters
from Utils.Network_functions import initialize_model
from Prepare_Data import Prepare_DataLoaders
from Utils.RBFHistogramPooling import HistogramLayer
from Utils.Confusion_mats import plot_confusion_matrix, plot_avg_confusion_matrix
from Utils.Generate_Learning_Curves import Plot_Learning_Curves
from Utils.Save_Results import get_file_location
import pdb

plt.ioff()

def main(Params):

    # Location of experimental results
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fig_size = Params['fig_size']
    font_size = Params['font_size']
    
    # Set up number of runs and class/plots names
    NumRuns = Params['Splits'][Params['Dataset']]
    plot_name = Params['Dataset'] + ' Test Confusion Matrix'
    avg_plot_name = Params['Dataset'] + ' Test Average Confusion Matrix'
    class_names = Class_names[Params['Dataset']]
    
    # Name of dataset
    Dataset = Params['Dataset']
    
    # Model(s) to be used
    model_name = Params['Model_name']
    
    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]
    
    # Number of bins and input convolution feature maps after channel-wise pooling
    numBins = Params['numBins']
    num_feature_maps = Params['out_channels'][model_name]
    
    # Local area of feature map after histogram layer
    feat_map_size = Params['feat_map_size']
    
    # Initialize arrays for results
    cm_stack = np.zeros((len(class_names), len(class_names)))
    cm_stats = np.zeros((len(class_names), len(class_names), NumRuns))
    FDR_scores = np.zeros((len(class_names), NumRuns))
    log_FDR_scores = np.zeros((len(class_names), NumRuns))
    accuracy = np.zeros(NumRuns)
    MCC = np.zeros(NumRuns)
    
    for split in range(0, NumRuns):
        
       
        #Find directory of results
        sub_dir = get_file_location(Params,split)
        
        # Load training and testing files (Python)
        train_pkl_file = open(sub_dir + 'train_dict.pkl', 'rb')
        train_dict = pickle.load(train_pkl_file)
        train_pkl_file.close()
    
        test_pkl_file = open(sub_dir + 'test_dict.pkl', 'rb')
        test_dict = pickle.load(test_pkl_file)
        test_pkl_file.close()
    
        # Remove pickle files
        del train_pkl_file, test_pkl_file
    
        # #Load model
        histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
                                          Params['kernel_size'][model_name],
                                          num_bins=numBins, stride=Params['stride'],
                                          normalize_count=Params['normalize_count'],
                                          normalize_bins=Params['normalize_bins'])
    
        # Initialize the histogram model for this run
        model, input_size, feature_extraction_layer = initialize_model(model_name, num_classes,
                                              Params['in_channels'][model_name],
                                              num_feature_maps,
                                              feature_extract=Params['feature_extraction'],
                                              histogram=Params['histogram'],
                                              histogram_layer=histogram_layer,
                                              parallel=Params['parallel'],
                                              use_pretrained=Params['use_pretrained'],
                                              add_bn=Params['add_bn'],
                                              scale=Params['scale'],
                                              feat_map_size=feat_map_size,
                                              TDNN_feats=(Params['TDNN_feats'][Dataset] * len(Params['feature'])), input_features = Params['feature'])
    
        # Set device to cpu or gpu (if available)
        device_loc = torch.device(device)
    
        # Generate learning curves
        Plot_Learning_Curves(train_dict['train_acc_track'],
                             train_dict['train_error_track'],
                             train_dict['val_acc_track'],
                             train_dict['val_error_track'],
                             train_dict['best_epoch'],
                             sub_dir)
    
        # If parallelized, need to set change model
        if Params['Parallelize']:
            model = nn.DataParallel(model)
    
        # model.load_state_dict(train_dict['best_model_wts'])
        print('Loading model...')
        model.load_state_dict(torch.load(sub_dir + 'Best_Weights.pt', map_location=device_loc))
        model = model.to(device)
        feature_extraction_layer = feature_extraction_layer.to(device)

    
        dataloaders_dict = Prepare_DataLoaders(Params)
    
        if (Params['TSNE_visual']):
            print("Initializing Datasets and Dataloaders...")
    
            dataloaders_dict = Prepare_DataLoaders(Params)
            print('Creating TSNE Visual...')
            
            #Remove fully connected layer
            if Params['Parallelize']:
                try:
                    model.module.fc = nn.Sequential()
                except:
                    model.module.classifier = nn.Sequential()
            else:
                try:
                    model.fc = nn.Sequential()
                except:
                    model.classifier = nn.Sequential()
            # Generate TSNE visual
            FDR_scores[:, split], log_FDR_scores[:, split] = Generate_TSNE_visual(
                dataloaders_dict,
                model, sub_dir, device, class_names,
                histogram=Params['histogram'],
                Separate_TSNE=Params['TSNE_visual'], input_features=Params['feature'], feature_layer=feature_extraction_layer)
            
        # Create CM for testing data
        cm = confusion_matrix(test_dict['GT'], test_dict['Predictions'])
        
        
        # Create classification report
        report = classification_report(test_dict['GT'], test_dict['Predictions'],
                                       target_names=class_names, output_dict=True)

        
        # Convert to dataframe and save as .CSV file
        df = pd.DataFrame(report).transpose()
        
        # Save to CSV
        df.to_csv((sub_dir + 'Classification_Report.csv'))
    
        # Confusion Matrix
        np.set_printoptions(precision=2)
        fig4, ax4 = plt.subplots(figsize=(fig_size, fig_size))
        plot_confusion_matrix(cm, classes=class_names, title=plot_name, ax=ax4,
                              fontsize=font_size)
        fig4.savefig((sub_dir + 'Confusion Matrix.png'), dpi=fig4.dpi)
        plt.close(fig4)
        cm_stack = cm + cm_stack
        cm_stats[:, :, split] = cm
    
        # Get accuracy of each cm
        accuracy[split] = 100 * sum(np.diagonal(cm)) / sum(sum(cm))
        # Write to text file
        with open((sub_dir + 'Accuracy.txt'), "w") as output:
            output.write(str(accuracy[split]))
    
        # Compute Matthews correlation coefficient
        MCC[split] = matthews_corrcoef(test_dict['GT'], test_dict['Predictions'])
    
        # Write to text file
        with open((sub_dir + 'MCC.txt'), "w") as output:
            output.write(str(MCC[split]))
        directory = os.path.dirname(os.path.dirname(sub_dir)) + '/'
    
        print('**********Run ' + str(split + 1) + ' Finished**********')
    
    directory = os.path.dirname(os.path.dirname(sub_dir)) + '/'
    np.set_printoptions(precision=2)
    fig5, ax5 = plt.subplots(figsize=(fig_size, fig_size))
    plot_avg_confusion_matrix(cm_stats, classes=class_names,
                              title=avg_plot_name, ax=ax5, fontsize=font_size)
    fig5.savefig((directory + 'Average Confusion Matrix.png'), dpi=fig5.dpi)
    plt.close()
    
    # Write to text file
    with open((directory + 'Overall_Accuracy.txt'), "w") as output:
        output.write('Average accuracy: ' + str(np.mean(accuracy)) + ' Std: ' + str(np.std(accuracy)))
    
    # Write to text file
    with open((directory + 'Overall_MCC.txt'), "w") as output:
        output.write('Average MCC: ' + str(np.mean(MCC)) + ' Std: ' + str(np.std(MCC)))
    
    # Write to text file
    with open((directory + 'testing_Overall_FDR.txt'), "w") as output:
        output.write('Average FDR: ' + str(np.mean(FDR_scores, axis=1))
                     + ' Std: ' + str(np.std(FDR_scores, axis=1)))
    with open((directory + 'test_Overall_Log_FDR.txt'), "w") as output:
        output.write('Average FDR: ' + str(np.mean(log_FDR_scores, axis=1))
                     + ' Std: ' + str(np.std(log_FDR_scores, axis=1)))
    
    # Write list of accuracies and MCC for analysis
    np.savetxt((directory + 'List_Accuracy.txt'), accuracy.reshape(-1, 1), fmt='%.2f')
    np.savetxt((directory + 'List_MCC.txt'), MCC.reshape(-1, 1), fmt='%.2f')
    
    np.savetxt((directory + 'test_List_FDR_scores.txt'), FDR_scores, fmt='%.2E')
    np.savetxt((directory + 'test_List_log_FDR_scores.txt'), log_FDR_scores, fmt='%.2f')
    plt.close("all")

def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments for dataset')
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                        help='Save results of experiments(default: True)')
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
    parser.add_argument('--train_batch_size', type=int, default=256,
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
                        help='learning rate (default: 0.01)')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='sigma for toy dataset (default: 0.1)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
    parser.add_argument('--audio_feature', nargs='+', default=['STFT'],
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
