#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 17:51:34 2021

@author: sohaas
"""

import unittest
import pandas as pd
from code.feature_extraction.character_length import CharacterLength

class CharacterLengthTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.character_length = CharacterLength(self.INPUT_COLUMN)
    
    def test_input_columns(self):
        self.assertEqual(self.character_length._input_columns, [self.INPUT_COLUMN])
        
    def test_feature_name(self):
        self.assertEqual(self.character_length.get_feature_name(), self.INPUT_COLUMN + "_charlength")

    def test_sentiment(self):
        input_text = "This tweets contains 34 characters"
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        charLength = self.character_length.fit_transform(input_df)
        EXPECTED_CHAR_LENGTH = [34]
        
        self.assertEqual(charLength, EXPECTED_CHAR_LENGTH)
        

if __name__ == '__main__':
    unittest.main()# -*- coding: utf-8 -*-

