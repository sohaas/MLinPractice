#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 09:46:18 2021

@author: tjweber
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class Topics(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__(input_column, "{0}_topics".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the word length based on the inputs
    def _get_values(self, inputs):
        
        topic_features = []
        for col in inputs:
            topic_features.append(np.array(col).reshape(-1,1))
        return topic_features
        
        """result = np.array(inputs[0].str.len())
        result = result.reshape(-1,1)
        return result"""
