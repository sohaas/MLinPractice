#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove inflections and map the tokenized words of the tweet to their root form.
Attention: Must only be applied on a "_tokenized" column!

Created on Thu Oct  7 15:12:31 2021

@author: sohaas
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.stem import WordNetLemmatizer
import ast

class Lemmatizer(Preprocessor):
    """Lemmatizes the words from the given input column into their root form."""
    
    def __init__(self, input_column, output_column):
        """Initialize the Lemmatizer with the given input and output column."""
        super().__init__([input_column], output_column)
    
    # don't need to implement _set_variables(), since no variables to set
    
    def _get_values(self, inputs, df):
        """Lemmatize the tweet."""
        
        print("Lemmatizing")
        lemmatized = []
        lemmatizer = WordNetLemmatizer()
        
        for index, value in inputs[0].items():
            tokenized_list = ast.literal_eval(value)
            lemmatized_tweet = []
            for word in tokenized_list:
                lemmatized_word = lemmatizer.lemmatize(word)
                lemmatized_tweet.append(lemmatized_word)
                
            lemmatized.append(str(lemmatized_tweet))
        
        return lemmatized
