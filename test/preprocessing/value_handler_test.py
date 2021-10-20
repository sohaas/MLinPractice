#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:41:06 2021

@author: tjweber
"""

import unittest
import pandas as pd
import pandas.testing as pd_testing
from code.preprocessing.value_handler import ValueHandler

class ValueHandlerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = None
        self.OUTPUT_COLUMN = None
        self.value_handler = ValueHandler(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
        
    def test_input_columns(self):
        self.assertListEqual(self.value_handler._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.value_handler._output_column, self.OUTPUT_COLUMN)
        
    def test_value_handling(self):
        input_data = [["test", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                      ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                      [" ", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                      [None, "", "", "", "", "", "", "", "", "", "", "", "", ""]]
        input_df = pd.DataFrame(
            data=input_data, 
            columns=["tweet", "place", "cashtags", "retweet", "near", "geo", 
                     "source", "user_rt_id", "user_rt", "retweet_id", 
                     "retweet_date", "translate", "trans_src", "trans_dest"])
        output_df = pd.DataFrame(data=[["test"]], columns=["tweet"])
        
        value_handled = self.value_handler.fit_transform(input_df)
        self.assertDataframeEqual(value_handled, output_df)
    
    def assertDataframeEqual(self, first, second):
        try:
            pd_testing.assert_frame_equal(first, second)
        except AssertionError as e:
            raise self.failureException() from e
            
            
if __name__ == '__main__':
    unittest.main()# -*- coding: utf-8 -*-
        
    
