#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:41:13 2021

@author: tjweber
"""

import unittest
import pandas as pd
from code.preprocessing.topic_extractor import TopicExtractor

class TopicExtractorTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN_TWEET = "input tweet"
        self.INPUT_COLUMN_LABLE = "input lable"
        self.OUTPUT_COLUMN = None
        self.topic_extractor = TopicExtractor([self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_LABLE], self.OUTPUT_COLUMN)
        
    def test_input_columns(self):
        self.assertListEqual(self.topic_extractor._input_columns, [self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_LABLE])

    def test_output_column(self):
        self.assertEqual(self.topic_extractor._output_column, self.OUTPUT_COLUMN)
        
    def test_topic_generation(self):
        input_data = [["['example']", True], ["['tweets']", True],
                      ["['example']", True], ["['tweets']", True], 
                      ["['example']", True], ["['tweets']", True],
                      ["['test']", False], ["['test']", False], 
                      ["['test']", False], ["['check']", True], 
                      ["['check']", True]]
        input_df = pd.DataFrame(data=input_data, columns=[self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_LABLE])
        output_columns = [self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_LABLE, "topic_example", "topic_tweets"]

        generated = self.topic_extractor.fit_transform(input_df)    
        self.assertListEqual(list(generated.columns), output_columns)
        
    def test_topic_extraction(self):
        input_data = [["['example']", True], ["['tweets']", True],
                      ["['test']", False], ["['example']", True], 
                      ["['tweets']", True], ["['example']", True], 
                      ["['tweets']", True]]
        input_df = pd.DataFrame(data=input_data, columns=[self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_LABLE])
        output_column = "topic_example"
        output_topic = [True, False, False, True, False, True, False]
        
        extracted = self.topic_extractor.fit_transform(input_df) 
        self.assertListEqual(list(extracted[output_column]), output_topic)
        
    def test_topic_limit(self):
        input_data = [["['hakuna']", True], ["['matata']", True], 
                      ["['what']", True], ["['a']", True], 
                      ["['wonderful']", True], ["['phrase']", True],
                      ["['hakuna']", True], ["['matata']", True], 
                      ["['aint']", True], ["['no']", True], ["['passing']", True],
                      ["['craze']", True], ["['it']", True], ["['means']", True],
                      ["['no']", True], ["['worries']", True], ["['for']", True],
                      ["['the']", True], ["['rest']", True], ["['of']", True],
                      ["['your']", True], ["['days']", True], ["['its']", True],
                      ["['our']", True], ["['problem-free']", True], 
                      ["['philosophy']", True],
                      ["['hakuna']", True], ["['matata']", True], 
                      ["['what']", True], ["['a']", True], 
                      ["['wonderful']", True], ["['phrase']", True],
                      ["['hakuna']", True], ["['matata']", True], 
                      ["['aint']", True], ["['no']", True], ["['passing']", True],
                      ["['craze']", True], ["['it']", True], ["['means']", True],
                      ["['no']", True], ["['worries']", True], ["['for']", True],
                      ["['the']", True], ["['rest']", True], ["['of']", True],
                      ["['your']", True], ["['days']", True], ["['its']", True],
                      ["['our']", True], ["['problem-free']", True], 
                      ["['philosophy']", True],
                      ["['hakuna']", True], ["['matata']", True], 
                      ["['what']", True], ["['a']", True], 
                      ["['wonderful']", True], ["['phrase']", True],
                      ["['hakuna']", True], ["['matata']", True], 
                      ["['aint']", True], ["['no']", True], ["['passing']", True],
                      ["['craze']", True], ["['it']", True], ["['means']", True],
                      ["['no']", True], ["['worries']", True], ["['for']", True],
                      ["['the']", True], ["['rest']", True], ["['of']", True],
                      ["['your']", True], ["['days']", True], ["['its']", True],
                      ["['our']", True], ["['problem-free']", True], 
                      ["['philosophy']", True]]
        input_df = pd.DataFrame(data=input_data, columns=[self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_LABLE])
        
        extracted = self.topic_extractor.fit_transform(input_df) 
        self.assertEqual(extracted.shape, (78, 12))
        
    def test_synonym_recognition(self):
        input_data = [["['test']", True], ["['test']", True], ["['test']", True],
                     ["['exam']", True], ["['trial']", True], 
                     ["['hakuna matata']", True]]
        input_df = pd.DataFrame(data=input_data, columns=[self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_LABLE])
        output_topic = [True, True, True, True, True, False]
        output_column = "topic_test"

        extracted = self.topic_extractor.fit_transform(input_df)    
        self.assertListEqual(list(extracted[output_column]), output_topic)        
    
    def test_tfidf_keywords(self):
        input_data = [["['hakuna matata']", True], ["['hakuna matata']", True],
                      ["['hakuna matata']", True], ["['hakuna']", True]]
        input_df = pd.DataFrame(data=input_data, columns=[self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_LABLE])
        output_columns = [self.INPUT_COLUMN_TWEET, self.INPUT_COLUMN_LABLE, "topic_matata"]

        generated = self.topic_extractor.fit_transform(input_df)    
        self.assertListEqual(list(generated.columns), output_columns)

        
if __name__ == '__main__':
    unittest.main()# -*- coding: utf-8 -*-
            