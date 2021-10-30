#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that converts the original tweet text to lowercase.

Created on Wed Oct  7 12:43:42 2021

@author: sohaas
"""

from code.preprocessing.preprocessor import Preprocessor

# converts the original tweet to lowercase
class Lowercaser(Preprocessor):
    
    # initialize the Lowercaser with the given input and output column
    def __init__(self, input_column, output_column):
        super().__init__([input_column], output_column)
    
    # don't need to implement _set_variables()
    
    # lowercase the tweet
    def _get_values(self, inputs, df):
        print("Lowercasing")
        lowercased = []
        
        for tweet in inputs[0]:                
            lowercased.append(tweet.lower())
        
        return lowercased