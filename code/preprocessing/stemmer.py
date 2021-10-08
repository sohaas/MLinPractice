#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove stems from the tokenized words of the tweet.
Attention: Must only be applied on a "_tokenized" column!

Created on Fri Oct  8 10:09:23 2021

@author: sohaas
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.stem import SnowballStemmer
import ast

class Stemmer(Preprocessor):
    """Removes the stems from the words of the given input column."""
    
    def __init__(self, input_column, output_column):
        """Initialize the Stemmer with the given input and output column."""
        super().__init__([input_column], output_column)
    
    # don't need to implement _set_variables(), since no variables to set
    
    def _get_values(self, inputs):
        """Stem the tweet."""
        
        stemmed = []
        stemmer = SnowballStemmer("english")
        
        for index, value in inputs[0].items():
            tokenized_list = ast.literal_eval(value)
            stemmed_tweet = []
            for word in tokenized_list:
                stemmed_word = stemmer.stem(word)
                stemmed_tweet.append(stemmed_word)
                
            stemmed.append(str(stemmed_tweet))
        
        return stemmed
# -*- coding: utf-8 -*-

