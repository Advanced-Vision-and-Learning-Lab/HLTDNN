#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:14:01 2023
Define optimizer for training model
@author: jarin.ritu
"""
import torch.optim as optim

def get_optimizer(params,optimizer_name,lr):
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(params,lr=lr)
        
    elif optimizer_name == 'Adamax':
        optimizer = optim.Adamax(params,lr=lr)
        
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(params,lr=lr)
        
    else:
        raise RuntimeError('{} not implemented'.format(optimizer_name))
    
    return optimizer