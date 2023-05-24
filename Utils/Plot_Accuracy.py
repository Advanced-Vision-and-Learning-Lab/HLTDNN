# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:52:34 2022

@author: jpeeples
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle
from matplotlib.pyplot import cm
import os
import pdb

def plot_metrics_multiple_runs(all_results, plot_title, dim,
                               ax = None,fig_dir='Metrics/'):
    """
    ******************************************************************
        *
        *  Func:   plot_roc_multiple_runs(all_results, plot_title)
        *
        *  Desc:   Plots a ROC curve for the given target and predicted labels.
        *
        *  Inputs:
        *          all_results - list of dictionaries containing:
        *                        all_fpr, all_tpr, all_auc, all_algs
        *
        *          plot_title - string denoting title of the generated plot
        *
        *  Outputs:
        *          None
        *
    ******************************************************************
    """

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    metric = 'total_loss'
    count = 0
    for phase in ['train','val']:
        fig, ax = plt.subplots(figsize=(7,7))
        for key in all_results.keys():
            
            mean_loss = np.mean(all_results[key][phase][metric], axis=0)
            std_loss = np.std(all_results[key][phase][metric], axis=0)
            
        ##################### Plot the learning curves ######################
            lw=2
          
            ax.plot(mean_loss, lw=lw,label='{} = {}'.format(r'$\lambda$',key))
            
            # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2, label=r'$\pm$ 1 std. dev.')
            ax.fill_between(np.linspace(0, len(mean_loss), len(mean_loss)), 
                            mean_loss + std_loss, 
                            mean_loss-std_loss, alpha=.2)
         
        if phase == 'val':
            phase = 'Validation'
        ax.set_title('{}, {}'.format(phase.capitalize() ,plot_title),fontsize=14)
        ax.set_xlabel('Epochs',fontsize=12)
        ax.set_ylabel('Loss',fontsize=12)
        # ax.set_aspect('equal', adjustable='box')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
        # Put a legend to the right of the current axis
        ax.set_ylim([0, 4.5])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
        # plt.tight_layout()
        fig.savefig('{}{}_{}_{}D'.format(fig_dir,phase,metric,dim))
        plt.close()
        
        count += 1
    
    return