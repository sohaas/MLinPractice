#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 17:51:34 2021

@author: sohaas
"""

import unittest
import pandas as pd
import numpy as np
from code.feature_extraction.sentiment_analyzer import SentimentAnalyzer

class SentimentAnalyzerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.sentiment_analyzer = SentimentAnalyzer(self.INPUT_COLUMN)
    
    def test_input_columns(self):
        self.assertEqual(self.sentiment_analyzer._input_columns, [self.INPUT_COLUMN])
        
    def test_feature_name(self):
        self.assertEqual(self.sentiment_analyzer.get_feature_name(), self.INPUT_COLUMN + "_sentiment")

    def test_sentiment(self):
        input_text_pos = "This is a very bad tweet containing evil thoughts"
        input_text_neg = "This is a great tweet that spreads happiness"
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text_pos, input_text_neg]
        
        sentiment = self.sentiment_analyzer.fit_transform(input_df)
        POS = [1, 0]
        NEG = [0, 1]
        EXPECTED_SENTIMENT = np.array([POS, NEG])
        
        self.assertEqual(sentiment.all(), EXPECTED_SENTIMENT.all())
        

if __name__ == '__main__':
    unittest.main()