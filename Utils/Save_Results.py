# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:39:28 2020
Save results from training/testing model
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import os
import pickle

## PyTorch dependencies
import torch


def get_file_location(Network_parameters,split):
    
    #Save audio feature datasets based on features
    if (Network_parameters['histogram']):
        
        if Network_parameters['audio_features']:
            if (Network_parameters['parallel']):
                filename = (Network_parameters['folder'] + '/' + Network_parameters['mode']
                            + '/' + Network_parameters['Dataset'] + '/'
                            + Network_parameters['hist_model'] + '/Parallel/' 
                            + Network_parameters['feature'] +'/' 'Run_' + str(split + 1) + '/')
            else:
                filename = (Network_parameters['folder'] + '/' + Network_parameters['mode']
                            + '/' + Network_parameters['Dataset'] + '/'
                            + Network_parameters['hist_model'] + '/' + 
                            Network_parameters['feature'] +'/' + 'Run_' + str(split + 1) + '/')
            
        else:
            if (Network_parameters['parallel']):
                filename = (Network_parameters['folder'] + '/' + Network_parameters['mode']
                            + '/' + Network_parameters['Dataset'] + '/'
                            + Network_parameters['hist_model'] + '/Parallel/Run_' + str(split + 1) + '/')
            else:
                filename = (Network_parameters['folder'] + '/' + Network_parameters['mode']
                            + '/' + Network_parameters['Dataset'] + '/'
                            + Network_parameters['hist_model'] + '/Run_' + str(split + 1) + '/')
    # Baseline model
    else:
        if Network_parameters['audio_features']:
            
            filename = (Network_parameters['folder'] + '/' + Network_parameters['mode']
                        + '/' + Network_parameters['Dataset'] + '/GAP_' +
                        Network_parameters['Model_name'] + '/' + Network_parameters['feature'] +'/'
                        + '/Run_' + str(split + 1) + '/')
        else:
            
            filename = (Network_parameters['folder'] + '/' + Network_parameters['mode']
                        + '/' + Network_parameters['Dataset'] + '/GAP_' +
                        Network_parameters['Model_name']
                        + '/Run_' + str(split + 1) + '/')
            
    return filename
    
    
def save_results(train_dict, test_dict, split, Network_parameters, num_params):
   
    filename = get_file_location(Network_parameters,split)
    
    if not os.path.exists(filename):
        os.makedirs(filename)
        
    #Will need to update code to save everything except model weights to
    # dictionary (use torch save)
    #Save training and testing dictionary, save model using torch
    torch.save(train_dict['best_model_wts'], filename + 'Best_Weights.pt')
    #Remove model from training dictionary
    train_dict.pop('best_model_wts')
    output_train = open(filename + 'train_dict.pkl','wb')
    pickle.dump(train_dict,output_train)
    output_train.close()
    
    output_test = open(filename + 'test_dict.pkl','wb')
    pickle.dump(test_dict,output_test)
    output_test.close()
    
    with open((filename + 'Test_Accuracy.txt'), "w") as output:
        output.write(str(test_dict['test_acc']))
    if not os.path.exists(filename):
        os.makedirs(filename)
    with open((filename + 'Num_parameters.txt'), "w") as output:
        output.write(str(num_params))