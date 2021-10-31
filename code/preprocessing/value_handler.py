#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:41:29 2021

@author: tjweber
"""

from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMNS_REMOVE

# handles missing and faulty values in the input columns
class ValueHandler(Preprocessor):
    
    # initialize the Tokenizer with the given input and output column
    def __init__(self, input_column, output_column):
        super().__init__([input_column], output_column)
    
    # store columns to be removed
    def _set_variables(self, inputs):
        self.rem_columns = COLUMNS_REMOVE
    
    # delete unnecessary data
    def _get_values(self, inputs, df):
        print("Handling missing values")
        # delete columns with no or an unsufficient amount of values
        df.columns = [x.replace("\r", "") for x in df.columns.to_list()]
        for col in self.rem_columns:
            del df[col]
            
        # delete rows without tweet
        empty_tweet = df["tweet"]!=("" or " " or None)
        df = df[empty_tweet]
            
        return df