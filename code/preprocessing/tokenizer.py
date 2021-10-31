#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenize the tweet into individual words.

Created on Wed Oct  6 13:59:54 2021

@author: lbechberger
"""

from code.preprocessing.preprocessor import Preprocessor
import nltk

# tokenizes the given input column into individual words
class Tokenizer(Preprocessor):
    
    # initialize the Tokenizer with the given input and output column
    def __init__(self, input_column, output_column):
        super().__init__([input_column], output_column)
    
    # don't need to implement _set_variables(), since no variables to set
    
    # tokenize the tweet
    def _get_values(self, inputs, df):
        print("Tokenizing")
        tokenized = []
        
        for tweet in inputs[0]:
            sentences = nltk.sent_tokenize(tweet)
            tokenized_tweet = []
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
                tokenized_tweet += words
            
            tokenized.append(str(tokenized_tweet))
        
        return tokenized
