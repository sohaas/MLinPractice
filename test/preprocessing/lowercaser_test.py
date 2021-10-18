#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 13:39:52 2021

@author: sohaas
"""

import unittest
import pandas as pd
from code.preprocessing.lowercaser import Lowercaser

class LowercaserTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.lowercaser = Lowercaser(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
    
    def test_input_columns(self):
        self.assertListEqual(self.lowercaser._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.lowercaser._output_column, self.OUTPUT_COLUMN)
        
    def test_value_handling(self):
        input_text = "ThIs iS AN eXaMpLE seNTeNce"
        output_text = "this is an example sentence"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        lowercased = self.lowercaser.fit_transform(input_df)
        self.assertEqual(lowercased[self.OUTPUT_COLUMN][0], output_text)
    

if __name__ == '__main__':
    unittest.main()