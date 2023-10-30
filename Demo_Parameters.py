# -*- coding: utf-8 -*-
"""
Parameters for histogram layer experiments
Only change parameters in this file before running
demo.py
@author: jpeeples 
"""
import os
import sys

def Parameters(args):
    ######## ONLY CHANGE PARAMETERS BELOW ########
    #Flag for if results are to be saved out
    #Set to True to save results out and False to not save results
    save_results = args.save_results
    
    #Location to store trained models
    #Always add slash (/) after folder name
    folder = args.folder
    
    #optimizer selection
    optimizer = args.optimizer
    
    #Flag to use histogram model or baseline global average pooling (GAP)
    # Set to True to use histogram layer and False to use GAP model
    histogram = args.histogram
    
    #Select dataset. Set to number of desired dataset
    data_selection = args.data_selection
    Dataset_names = {0: 'DeepShip'}
    
    #Number of bins for histogram layer. Recommended values are 4, 8 and 16.
    #Set number of bins to powers of 2 (e.g., 2, 4, 8, etc.)
    #For HistRes_B models using ResNet18 and ResNet50, do not set number of bins
    #higher than 128 and 512 respectively. Note: a 1x1xK convolution is used to
    #downsample feature maps before binning process. If the bin values are set
    #higher than 128 or 512 for HistRes_B models using ResNet18 or ResNet50
    #respectively, than an error will occur due to attempting to reduce the number of
    #features maps to values less than one
    numBins = args.numBins
    
    
    #Flag for feature extraction. False, train whole model. True, only update
    #fully connected and histogram layers parameters (default: False)
    #Flag to use pretrained model from ImageNet or train from scratch (default: True)
    #Flag to add BN to convolutional features (default:True)
    #Location/Scale at which to apply histogram layer (default: 5 (at the end))
    feature_extraction = args.feature_extraction
    use_pretrained = args.use_pretrained
    add_bn = True
    scale = 5
    
    #Set learning rate for new layers
    #Recommended values are .01 (used in paper) or .001
    lr = args.lr
    
    #Parameters of Histogram Layer
    #For no padding, set 0. If padding is desired,
    #enter amount of zero padding to add to each side of image
    #(did not use padding in paper, recommended value is 0 for padding)
    padding = 0
    
    #Reduce dimensionality based on number of output feature maps from GAP layer
    #Used to compute number of features from histogram layer
    out_channels = {"resnet50": 2048, "resnet18": 512, "efficientnet": 1280, 
                    "resnet50_wide": 2048, "resnet50_next": 2048, "densenet121": 4096,
                    "regnet": 400, "TDNN": 256,"DNN": 256}
    
    #Set whether to have the histogram layer inline or parallel (default: parallel)
    #Set whether to use sum (unnormalized count) or average pooling (normalized count)
    # (default: average pooling)
    #Set whether to enforce sum to one constraint across bins (default: True)
    parallel = True
    normalize_count = True
    normalize_bins = True
    
    #Set step_size and decay rate for scheduler
    #In paper, learning rate was decayed factor of .1 every ten epochs (recommended)
    step_size = 10
    gamma = .1
    
    #Batch size for training and epochs. If running experiments on single GPU (e.g., 2080ti),
    #training batch size is recommended to be 64. If using at least two GPUs,
    #the recommended training batch size is 128 (as done in paper)
    #May need to reduce batch size if CUDA out of memory issue occurs
    batch_size = {'train': args.train_batch_size, 'val': args.val_batch_size, 'test': args.test_batch_size} 
    num_epochs = args.num_epochs
    
    #Resize the image before center crop. Recommended values for resize is 256 (used in paper), 384,
    #and 512 (from http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf)
    #Center crop size is recommended to be 256.
    # resize_size = args.resize_size
    # center_size = 224
    
    #Pin memory for dataloader (set to True for experiments)
    pin_memory = False
    
    #Set number of workers, i.e., how many subprocesses to use for data loading.
    #Usually set to 0 or 1. Can set to more if multiple machines are used.
    #Number of workers for experiments for two GPUs was three
    num_workers = 0
    
    #Output feature map size after histogram layer
    feat_map_size = 4
    
    #Select audio feature for DeepShip 
    feature = args.audio_feature
    
    #Set filter size and stride based on scale
    # Current values will produce 2x2 local feature maps
    # TODO: Switch to adaptive average pooling
    if scale == 1:
        stride = [32, 32]
        in_channels = {"resnet50": 64, "resnet18": 64, "efficientnet": 64, "regnet": 64,
                       "resnet50_wide": 64, "resnet50_next": 64, "densenet121": 64, 'TDNN': 16}
        kernel_size = {"resnet50": [64,64], "resnet18": [64,64], "efficientnet": [64,64], "regnet": [64,64],
                       "resnet50_wide": [64,64], "resnet50_next": [64,64], "densenet121": [64,64],'TDNN': [3,3]}
    elif scale == 2:
        stride = [16, 16]
        in_channels = {"resnet50": 256, "resnet18": 64, "efficientnet": 64, "regnet": 64,
                       "resnet50_wide": 64, "resnet50_next": 64, "densenet121": 64, 'DNN': 16}
        kernel_size = {"resnet50": [32,32],  "resnet18": [32,32], "efficientnet": [32,32], "regnet": [32,32],
                       "resnet50_wide": [32,32], "resnet50_next": [32,32], "densenet121": [32,32],'TDNN': [3,3]}
    elif scale == 3:
        stride = [8, 8]
        in_channels = {"resnet50": 512, "resnet18": 128, "efficientnet": 128, "regnet": 128,
                       "resnet50_wide": 512, "resnet50_next": 512, "densenet121": 128, 'DNN': 16}
        kernel_size = {"resnet50": [16,16],  "resnet18": [16,16], "efficientnet": [16,16], "regnet": [32,32],
                       "resnet50_wide": [16,16], "resnet50_next": [16,16], "densenet121": [16,16],'TDNN': [3,3]}
    elif scale == 4:
        stride = [4, 4]
        in_channels = {"resnet50": 1024, "resnet18": 256, "efficientnet": 256, "regnet": 256,
                       "resnet50_wide": 1024, "resnet50_next": 1024, "densenet121": 256, 'DNN': 4}
        kernel_size = {"resnet50": [8,8],  "resnet18": [8,8], "efficientnet": [16,16], "regnet": [64,64],
                       "resnet50_wide": [32,32], "resnet50_next": [32,32], "densenet121": [32,32],'TDNN': [3,3]}
    else:
        stride = [2, 2]
        in_channels = {"resnet50": 2048, "resnet18": 512, "efficientnet": 1280, "regnet": 400,
                       "resnet50_wide": 2048, "resnet50_next": 2048, "densenet121": 1024, 'TDNN': 4}
        kernel_size = {"resnet50": [4,4],  "resnet18": [4,4], "efficientnet": [4,4], "regnet": [4,4],
                       "resnet50_wide": [4,4], "resnet50_next": [4,4], "densenet121": [1,1],'TDNN': [3,3]}
    
    #Visualization of results parameters
    #Visualization parameters for figures
    fig_size = 12
    font_size = 24
    
    #Flag for TSNE visuals, set to True to create TSNE visual of features
    #Set to false to not generate TSNE visuals
    #Number of images to view for TSNE (defaults to all training imgs unless
    #value is less than total training images).
    TSNE_visual = True
    Num_TSNE_images = 10000
    
    #Set to True if more than one GPU was used
    Parallelize_model = True
    
    ######## ONLY CHANGE PARAMETERS ABOVE ########
    if feature_extraction:
        mode = 'Feature_Extraction'
    else:
        mode = 'Fine_Tuning'
    
    #Location of texture datasets
    Data_dirs = {'DeepShip': './Datasets/DeepShip/Segments/'}
    
    #ResNet models to use for each dataset
    Model_name = args.model
    
    #Number of classes in each dataset
    num_classes = {'DeepShip': 4}
    
    #Number of runs and/or splits for each dataset
    Splits = {'DeepShip': 3}
    
    #Number of input channels to TDNN
    #For audio features, this values will be 1 for TDNN 
    # (features are repeated three times along channel dimension for CNN models)
    TDNN_feats = {'DeepShip': 1}
    
    #Flag for whether or not a 
    
    Dataset = Dataset_names[data_selection]
    data_dir = Data_dirs[Dataset]
    
    #Save results based on features (can adapt for additional audio datasets or computer vision datasets)
    if (Dataset=='DeepShip'):
        audio_features = True
    else:
        audio_features = False
    
    
    Hist_model_name = 'Hist{}_{}'.format(Model_name,numBins)
    
    #Return dictionary of parameters
    Params = {'save_results': save_results,'folder': folder,
                          'histogram': histogram,'Dataset': Dataset, 'data_dir': data_dir,
                          'optimizer': optimizer,
                          'num_workers': num_workers, 'mode': mode,'lr': lr,
                          'step_size': step_size,
                          'gamma': gamma, 'batch_size' : batch_size, 
                          'num_epochs': num_epochs, 
                          'padding': padding, 
                          'stride': stride, 'kernel_size': kernel_size,
                          'in_channels': in_channels, 'out_channels': out_channels,
                          'normalize_count': normalize_count, 
                          'normalize_bins': normalize_bins,'parallel': parallel,
                          'numBins': numBins,'feat_map_size': feat_map_size,
                          'Model_name': Model_name, 'num_classes': num_classes, 
                          'Splits': Splits, 'feature_extraction': feature_extraction,
                          'hist_model': Hist_model_name, 'use_pretrained': use_pretrained,
                          'add_bn': add_bn, 'pin_memory': pin_memory, 'scale': scale,
                          'TSNE_visual': TSNE_visual,
                          'Parallelize': Parallelize_model,
                          'Num_TSNE_images': Num_TSNE_images,'fig_size': fig_size,
                          'font_size': font_size, 'feature': feature, 
                          'TDNN_feats': TDNN_feats, 'audio_features': audio_features}
    
    return Params
