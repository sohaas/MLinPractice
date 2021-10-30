#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 22:00:00 2021

@author: sohaas
"""

import unittest
import pandas as pd
from code.feature_extraction.url_included import UrlIncluded

class EnglishLanguageTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.url_included = UrlIncluded(self.INPUT_COLUMN)
    
    def test_input_columns(self):
        self.assertEqual(self.url_included._input_columns, [self.INPUT_COLUMN])
        
    def test_feature_name(self):
        self.assertEqual(self.url_included.get_feature_name(), self.INPUT_COLUMN + "_containsURL")

    def test_url(self):
        input_text = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        hasURL = self.url_included.fit_transform(input_df)
        # Contains URL = 1
        EXPECTED_OUTPUT = [1]
        
        self.assertEqual(hasURL, EXPECTED_OUTPUT)
        
    def test_no_url(self):
        input_text = "[]"
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        hasURL = self.url_included.fit_transform(input_df)
        # Does not contain URL = 0
        EXPECTED_OUTPUT = [0]
        
        self.assertEqual(hasURL, EXPECTED_OUTPUT)
        

if __name__ == '__main__':
    unittest.main()