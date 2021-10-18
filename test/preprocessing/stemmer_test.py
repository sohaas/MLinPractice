#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:19:30 2021

@author: sohaas
"""

import ast
import unittest
import pandas as pd
from code.preprocessing.stemmer import Stemmer

class StemmerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN_TWEET = "input tweet"
        self.INPUT_COLUMN_EN = "input language"
        self.OUTPUT_COLUMN = "output"
        self.stemmer = Stemmer([self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_EN], self.OUTPUT_COLUMN)
    
    def test_input_columns(self):
        self.assertListEqual(self.stemmer._input_columns, [self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_EN])

    def test_output_column(self):
        self.assertEqual(self.stemmer._output_column, self.OUTPUT_COLUMN)
       
    def test_stemming(self):
        input_tweet = "['This', 'is', 'an', 'example', 'sentence', 'for', 'showing', 'stemming']"
        input_language = "en"
        output_text = "['this', 'is', 'an', 'exampl', 'sentenc', 'for', 'show', 'stem']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN_TWEET] = [input_tweet]
        input_df[self.INPUT_COLUMN_EN] = [input_language]
        
        stemmed = self.stemmer.fit_transform(input_df)
        self.assertEqual(stemmed[self.OUTPUT_COLUMN][0], output_text)
    
    def test_equal_inflections(self):
        input_tweet = "['pass', 'passing', 'passed']"
        input_language = "en"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN_TWEET] = [input_tweet]
        input_df[self.INPUT_COLUMN_EN] = [input_language]
        
        stemmed = self.stemmer.fit_transform(input_df)
        stemmed_list = ast.literal_eval(stemmed[self.OUTPUT_COLUMN][0])

        self.assertEqual(stemmed_list[0], stemmed_list[1], stemmed_list[2])
        
    def test_unequal_inflections(self):
        input_tweet = "['car', 'caring', 'carrier']"
        input_language = "en"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN_TWEET] = [input_tweet]
        input_df[self.INPUT_COLUMN_EN] = [input_language]
        
        stemmed = self.stemmer.fit_transform(input_df)
        stemmed_list = ast.literal_eval(stemmed[self.OUTPUT_COLUMN][0])

        self.assertNotEqual(stemmed_list[0], stemmed_list[1], stemmed_list[2])
    

if __name__ == '__main__':
    unittest.main()# -*- coding: utf-8 -*-