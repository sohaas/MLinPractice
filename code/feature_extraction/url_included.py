#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that tells whether an URL is included in a tweet.

Created on Thu Sep 14 21:21:17 2021

@author: sohaas
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the url as a feature
class UrlIncluded(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_containsURL".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # check whether an url is included in the inputs
    def _get_values(self, inputs):
        
        has_url = []
        
        # encode using binary numbers to facilitate subsequent pipeline steps
        for urls in inputs[0]:
            if urls == "[]":
                has_url.append(0)
            else:          
                has_url.append(1)
                
        result = np.array(has_url) 
        result = result.reshape(-1,1)
        return result
