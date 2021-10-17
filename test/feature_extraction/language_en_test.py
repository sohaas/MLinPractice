#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 21:55:21 2021

@author: sohaas
"""

import unittest
import pandas as pd
from code.feature_extraction.language_en import EnglishLanguage

class EnglishLanguageTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.language_en = EnglishLanguage(self.INPUT_COLUMN)
    
    def test_input_columns(self):
        self.assertEqual(self.language_en._input_columns, [self.INPUT_COLUMN])
        
    def test_feature_name(self):
        self.assertEqual(self.language_en.get_feature_name(), self.INPUT_COLUMN + "_isEnglish")

    def test_english(self):
        input_text = "en"
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        isEnglish = self.language_en.fit_transform(input_df)
        # English = 1
        EXPECTED_OUTPUT = [1]
        
        self.assertEqual(isEnglish, EXPECTED_OUTPUT)
        
    def test_non_english(self):
        input_text = "fr"
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        isEnglish = self.language_en.fit_transform(input_df)
        # Non-English = 0
        EXPECTED_OUTPUT = [0]
        
        self.assertEqual(isEnglish, EXPECTED_OUTPUT)
        

if __name__ == '__main__':
    unittest.main()# -*- coding: utf-8 -*-

