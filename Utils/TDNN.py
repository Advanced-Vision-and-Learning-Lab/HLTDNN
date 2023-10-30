# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:27:21 2023
Baseline TDNN model
@author: jpeeples
"""

import torch
import torch.nn as nn
import pdb
#TBD: Create block of convolution, max pooling, and non-linearity
class TDNN(nn.Module):
    
    def __init__(self, in_channels, stride=1, dilation=1, batch_norm=True,
                num_class = 4, output_len = 1, drop_p = .1):
        '''
        Baseline TDNN model
       
        '''
        super(TDNN, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.dilation = dilation
        self.batch_norm = batch_norm
        self.output_len = output_len
        self.drop_p = drop_p
        
        
        #Define convolution layers
        self.conv1 = nn.Conv2d(self.in_channels, 16, kernel_size=(11,11),
                               padding='same',bias=True)
        self.conv2 = nn.Conv2d(16,16,kernel_size=(3,3),padding='same',bias=True)
        self.conv3 = nn.Conv2d(16,16,kernel_size=(3,3),padding='same',bias=True)
        self.conv4 = nn.Conv2d(16,4,kernel_size=(3,3),padding='same',bias=True)
        self.conv5 = nn.Conv1d(4,256,kernel_size=(1),padding='same',bias=True)
        
        #Define max pooling layers
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,2),stride=(1,1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1,4),stride=(1,2))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1,8),stride=(1,4))
        
        #Define nonlinearity 
        self.nonlinearity = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        #Define average pooling layer for desired length of signal
        self.avgpool = nn.AdaptiveAvgPool1d(self.output_len)
        
        #Add dropout if needed
        if drop_p is not None:
            self.dropout = nn.Dropout(p=self.drop_p)
        else:
            self.dropout = nn.Sequential()
            
        #Define classifier (fully connected layer)
        #Do not apply sigmoid, cross-entropy takes raw logits
        self.fc = nn.Linear(self.conv5.out_channels*self.output_len, num_class)
    
    def forward(self, x):
        '''
        input: size (batch, channels, audio_feature_x, audio_feature_y)
        output: size (batch, num_class)
        '''

        #Pass through feature extraction layers
        x = self.conv1(x)
        x = self.nonlinearity(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.nonlinearity(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.nonlinearity(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = self.nonlinearity(x)
        x = self.maxpool4(x)
        
        #Reshape to be N x C x (MxN)
        x = torch.flatten(x,start_dim=-2)
        
        #Apply last convolution filter, sigmoid, and average pool to desired length
        x = self.conv5(x)
        x = self.sigmoid(x)
        x = self.avgpool(x).flatten(start_dim=1)
        
        #Add dropout
        x = self.dropout(x)
        
        #Get classifier outputs for classes
        x = self.fc(x)
       
        return x
