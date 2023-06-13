# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:30:19 2023

@author: jpeeples
"""
import math
import pdb

def generate_spatial_dimensions(n):
    '''
    Code initialized using ChatGPT (Prompt: Create python function that finds two 
                                    integers that equal product)
    Modified to account for square outputs and to return (1,input value) 
    prime numbers
    Parameters
    ----------
    n : desired value for spatial resolution

    Raises
    ------
    ValueError
        Error raised if negative number

    Returns
    -------
    i : x dimension
    j : y dimension
    '''
    #If negative, return error
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
        
    #Compute root value
    root = math.sqrt(n)
    
    #Return same value for x and y if equal (square output) or
    #return non-square window
    #If prime number, return (1,n)
    if ((int(root + 0.5)**2) == n):
        return (int(root),int(root))
    else:
        items = []
        end_range = int(n ** 0.5) + 1
        for i in range(1, end_range):
            if n % i == 0:
                j = n // i
                items.append((j,i))
      
        return items[-1]
