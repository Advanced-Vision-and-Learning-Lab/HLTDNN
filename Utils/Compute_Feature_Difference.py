# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:48:39 2023

@author: jpeeples
"""

import numpy as np
import torch
from sklearn.metrics import calinski_harabasz_score as CH_score
from itertools import combinations
# from Utils.Compute_CH import clusterwise_calinski_harabasz_score
from Utils.Compute_FDR import Compute_Fisher_Score
from barbar import Bar


import matplotlib.pyplot as plt

def Compute_feature_matrix(dataloader, difference='CH', device='cpu', save_figure=False):
    # Set distance function
    if difference == 'CH':
        dist_func = CH_score
    elif difference == 'FDR':  # Default to FDR
        dist_func = Compute_Fisher_Score
    else:
        raise RuntimeError('{} not implemented'.format(difference))

    # Iterate through data and grab inputs
    GT_vals = np.array(0)
    saved_imgs = []
    for idx, (inputs, classes, index) in enumerate(Bar(dataloader)):
        images = inputs.to(device)
        labels = classes.to(device, torch.long)
        GT_vals = np.concatenate((GT_vals, labels.cpu().numpy()), axis=None)
        saved_imgs.append(images.cpu().flatten(start_dim=1).numpy())

    # Reshape saved images to be N x D (number of samples by features)
    saved_imgs = np.concatenate(saved_imgs, axis=0)

    # Compute difference scores scores
    GT_vals = GT_vals[1:]

    # Generate difference matrix
    classes = np.unique(GT_vals)
    diff_matrix = np.zeros((len(classes), len(classes)))

    # Get all combinations of two classes
    class_combos = combinations(classes, 2)

    for combo in class_combos:
        # Grab data that pertains to classes of interest
        class1_imgs = saved_imgs[GT_vals == combo[0]]
        class1_labels = GT_vals[GT_vals == combo[0]]
        class2_imgs = saved_imgs[GT_vals == combo[1]]
        class2_labels = GT_vals[GT_vals == combo[1]]

        # Compute distance between classes
        if difference == 'FDR':
            score = dist_func(np.concatenate((class1_imgs, class2_imgs), axis=0),
                              np.concatenate((class1_labels, class2_labels), axis=0))
            score = np.mean(score)  # Compute the mean if dist_func returns an array
        else:
            score = dist_func(np.concatenate((class1_imgs, class2_imgs), axis=0),
                              np.concatenate((class1_labels, class2_labels), axis=0))

        # Fill symmetric scores in matrix
        diff_matrix[combo[0], combo[1]] = score
        diff_matrix[combo[1], combo[0]] = score
    
    # Plot the FDR matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(diff_matrix, cmap='coolwarm')
    plt.title('Difference Matrix')
    plt.colorbar()
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xticks(rotation=45)
    
    # Save the figure if specified
    if save_figure:
        plt.savefig('difference_matrix.png')
    
    return diff_matrix, np.log(diff_matrix + np.eye(len(classes)))
