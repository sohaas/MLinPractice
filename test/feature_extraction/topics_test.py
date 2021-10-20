#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:54:17 2021

@author: tjweber
"""

import unittest
import pandas as pd
import numpy as np
import numpy.testing as np_testing
from code.feature_extraction.topics import Topics

class TopicFeatureTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMNS = ["topic_1"]
        self.topic_feature = Topics(self.INPUT_COLUMNS)
    
    def test_input_columns(self):
        self.assertListEqual(self.topic_feature._input_columns, self.INPUT_COLUMNS)
        
    def test_feature_name(self):
        self.assertEqual(self.topic_feature.get_feature_name(), "topics")
        
    def test_single_topic(self):
        input_topic = True
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMNS[0]] = [input_topic]
        
        topics = self.topic_feature.fit_transform(input_df)
        # Contains topic = 1
        EXPECTED_OUTPUT = [1]
        
        self.assertEqual(topics, EXPECTED_OUTPUT)
        
    def test_no_topic(self):
        input_topic = False
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMNS[0]] = [input_topic]
        
        topics = self.topic_feature.fit_transform(input_df)
        # Does not contain topic = 0
        EXPECTED_OUTPUT = [0]
        
        self.assertEqual(topics, EXPECTED_OUTPUT)
        
    def test_multiple_topics(self):
        self.INPUT_COLUMNS.append("topic_2")
        input_topic_1 = False
        input_topic_2 = True
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMNS[0]] = [input_topic_1]
        input_df[self.INPUT_COLUMNS[1]] = [input_topic_2]
        
        topics = self.topic_feature.fit_transform(input_df)
        # Contains topic = 1, does not contain topic = 0
        EXPECTED_OUTPUT = [[0], [1]]
        
        self.assertListEqual(topics, EXPECTED_OUTPUT) 
