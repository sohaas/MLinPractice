#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove stopwords from the tokenized words of the tweet.
Attention: Must only be applied on a "_tokenized" column!

Created on Fri Oct  8 12:17:46 2021

@author: sohaas
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.corpus import stopwords
import ast

class Stopworder(Preprocessor):
    """Removes the stopwords from the words of the given input column."""
    
    def __init__(self, input_column, output_column):
        """Initialize the Stopworder with the given input and output column."""
        super().__init__([input_column], output_column)
    
    # don't need to implement _set_variables(), since no variables to set
    
    def _get_values(self, inputs):
        """Remove stopwords."""
        
        stopword_list = set(stopwords.words("english"))
        no_stopwords = []
        
        for index, value in inputs[0].items():
            tokenized_list = ast.literal_eval(value)
            no_stopwords_tweet = []
            for word in tokenized_list:
                if word.lower() not in stopword_list:
                    no_stopwords_tweet.append(word)
                
            no_stopwords.append(str(no_stopwords_tweet))
        
        return no_stopwords

# -*- coding: utf-8 -*-

