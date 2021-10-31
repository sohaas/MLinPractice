#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that tells whether the language of a tweet is English or not.

Created on Thu Sep 14 17:47:42 2021

@author: sohaas
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the language as a feature
class EnglishLanguage(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_isEnglish".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # check whether the language of the inputs is English
    def _get_values(self, inputs):
        
        is_english = []
        
        # encode using binary numbers to facilitate subsequent pipeline steps
        for language in inputs[0]:
            if language == "en":
                is_english.append(1)
            else:          
                is_english.append(0)
        
        result = np.array(is_english) 
        result = result.reshape(-1,1)
        return result