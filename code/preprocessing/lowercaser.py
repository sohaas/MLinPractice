#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that converts the original tweet text to lowercase.

Created on Wed Oct  7 12:43:42 2021

@author: sohaas
"""

from code.preprocessing.preprocessor import Preprocessor
import nltk

class Lowercaser(Preprocessor):
    """Converts the original tweet to lowercase"""
    
    def __init__(self, input_column, output_column):
        """Initialize the Lowercaser with the given input and output column."""
        super().__init__([input_column], output_column)
    
    # don't need to implement _set_variables()
    
    def _get_values(self, inputs):
        """Lowercase the tweet."""
        lowercased = []
        
        for tweet in inputs[0]:                
            lowercased.append(tweet.lower())
        
        return lowercased
# -*- coding: utf-8 -*-
