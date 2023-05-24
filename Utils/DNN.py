# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:45:04 2023

@author: jpeeples
"""
import torch.nn as nn

#TBD: Create block of convolution, max pooling, and non-linearity
class DNN(nn.Module):
    
    def __init__(self, in_channels=768,num_class = 4, drop_p = 0):
        '''
        TDNN as defined by Model Description from MIT LL
       
        '''
        super(DNN, self).__init__()
        self.in_channels = in_channels
        self.drop_p = drop_p
        self.num_class = num_class
      
        #Define convolution layers
        self.input_layer = nn.Linear(in_channels, 602)
        self.hidden_layer_1 = nn.Linear(602, 256)
        self.hidden_layer_2 = nn.Linear(256, 64)
        self.hidden_layer_3 = nn.Linear(64, 16)
    
        #Define nonlinearity 
        self.nonlinearity = nn.ReLU()
        
        #Add dropout if needed
        if drop_p is not None:
            self.dropout = nn.Dropout(p=self.drop_p)
        else:
            self.dropout = nn.Sequential()
            
        #Define classifier (fully connected layer)
        #Do not apply sigmoid, cross-entropy takes raw logits
        self.fc = nn.Linear(self.hidden_layer_3.out_features, num_class)
    
    def forward(self, x):
        '''
        input: size (batch, channels, audio_feature_x, audio_feature_y)
        output: size (batch, num_class)
        '''

        #Pass through feature extraction layers
        #pdb.set_trace()
        x = x.flatten(start_dim=1)
        x = self.input_layer(x)
        x = self.nonlinearity(x)
        
        x = self.hidden_layer_1(x)
        x = self.nonlinearity(x)
        
        x = self.hidden_layer_2(x)
        x = self.nonlinearity(x)
        
        x = self.hidden_layer_3(x)
        x = self.nonlinearity(x)
        
        #Add dropout
        x = self.dropout(x)
        
        #Get classifier outputs for classes
        x = self.fc(x)
       
        return x