#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:12:13 2021

@author: sohaas
"""

import unittest
import pandas as pd
from code.preprocessing.links_remover import LinksRemover

class LinksRemoverTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.links_remover = LinksRemover(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
    
    def test_input_columns(self):
        self.assertListEqual(self.links_remover._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.links_remover._output_column, self.OUTPUT_COLUMN)
        
    def test_link_remover(self):
        input_text = "This is an expample tweet https://xkcd.com/2054"
        output_text = "This is an expample tweet "
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        no_links = self.links_remover.fit_transform(input_df)
        self.assertEqual(no_links[self.OUTPUT_COLUMN][0], output_text)
    

if __name__ == '__main__':
    unittest.main()