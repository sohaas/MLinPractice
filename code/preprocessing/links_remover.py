#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:30:12 2021

@author: tjweber
"""

from code.preprocessing.preprocessor import Preprocessor

# removes links from the original tweet
class LinksRemover(Preprocessor):
    
    # constructor
    def __init__(self, input_column, output_column):
        # input column "tweet", new output column
        super().__init__([input_column], output_column)
    
    # set internal variables based on input columns
    def _set_variables(self, inputs):
        # store regEx for link removal for later reference
        self._regEx = r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)"
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs, df):
        # replace link with empty string
        column = inputs[0].str.replace(self._regEx, "", regex=True)
        return column