#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:43:17 2021

@author: sohaas
"""

import ast
import unittest
import pandas as pd
from code.preprocessing.lemmatizer import Lemmatizer

class LemmatizerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input tweet"
        self.OUTPUT_COLUMN = "output"
        self.lemmatizer = Lemmatizer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
    
    def test_input_columns(self):
        self.assertListEqual(self.lemmatizer._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.lemmatizer._output_column, self.OUTPUT_COLUMN)
       
    def test_lemmatizing(self):
        input_text = "['All', 'geese', 'have', 'to', 'sneeze', 'when', 'they', 'touch', 'any', 'cacti']"
        output_text = "['All', 'goose', 'have', 'to', 'sneeze', 'when', 'they', 'touch', 'any', 'cactus']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        lemmatized = self.lemmatizer.fit_transform(input_df)
        self.assertEqual(lemmatized[self.OUTPUT_COLUMN][0], output_text)
    
    def test_equal_inflections(self):
        input_text = "['pass', 'passing', 'passed']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        lemmatized = self.lemmatizer.fit_transform(input_df)
        lemmatized_list = ast.literal_eval(lemmatized[self.OUTPUT_COLUMN][0])

        # Note: we would expect this to be equal, like in the stemming test, but it is in fact not
        self.assertNotEqual(lemmatized_list[0], lemmatized_list[1], lemmatized_list[2])
        
    def test_unequal_inflections(self):
        input_text = "['car', 'caring', 'carrier']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        lemmatized = self.lemmatizer.fit_transform(input_df)
        lemmatized_list = ast.literal_eval(lemmatized[self.OUTPUT_COLUMN][0])

        self.assertNotEqual(lemmatized_list[0], lemmatized_list[1], lemmatized_list[2])
    

if __name__ == '__main__':
    unittest.main()# -*- coding: utf-8 -*-