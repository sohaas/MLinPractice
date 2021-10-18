#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 13:58:03 2021

@author: sohaas
"""

import unittest
import pandas as pd
from code.preprocessing.stopworder import Stopworder

class StopworderTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN_TWEET = "input tweet"
        self.INPUT_COLUMN_EN = "input language"
        self.OUTPUT_COLUMN = "output"
        self.stopworder = Stopworder([self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_EN], self.OUTPUT_COLUMN)
    
    def test_input_columns(self):
        self.assertListEqual(self.stopworder._input_columns, [self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_EN])

    def test_output_column(self):
        self.assertEqual(self.stopworder._output_column, self.OUTPUT_COLUMN)
       
    def test_stopworder(self):
        input_tweet = "['This', 'is', 'an', 'example', 'sentence', 'with', 'stopwords']"
        input_language = "en"
        output_text = "['example', 'sentence', 'stopwords']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN_TWEET] = [input_tweet]
        input_df[self.INPUT_COLUMN_EN] = [input_language]
        
        no_stopwords = self.stopworder.fit_transform(input_df)
        self.assertEqual(no_stopwords[self.OUTPUT_COLUMN][0], output_text)
    

if __name__ == '__main__':
    unittest.main()# -*- coding: utf-8 -*-

