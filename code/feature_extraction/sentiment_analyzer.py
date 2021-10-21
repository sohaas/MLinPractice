#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature that categorizes the sentiment of a tweet into 'positive', 'neutral' or 'negative'.

Created on Thu Sep 14 10:40:13 2021

@author: sohaas
"""

import numpy as np
<<<<<<< Updated upstream
=======
from sklearn.preprocessing import OrdinalEncoder
>>>>>>> Stashed changes
from code.feature_extraction.feature_extractor import FeatureExtractor
from nltk.sentiment import SentimentIntensityAnalyzer

# class for extracting the sentiment as a feature
class SentimentAnalyzer(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_sentiment".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # analyze the sentiment based on the inputs
    def _get_values(self, inputs):
        
        sia = SentimentIntensityAnalyzer()
        sentiment = []
        
        negative = bin(0)
        neutral = bin(1)
        positive = bin(2)
        
        for tweet in inputs[0]: 
            compoundScore = sia.polarity_scores(tweet)["compound"]
            if compoundScore <= -0.05:
                sentiment.append(negative)
            elif compoundScore >= 0.05: 
                sentiment.append(positive)
            else:          
<<<<<<< Updated upstream
                sentiment.append(neutral)
=======
                sentiment.append("neutral")
            
        # ordinal encoding
        features = np.array(sentiment) 
        features = features.reshape(-1,1)
        encoder = OrdinalEncoder()
        encoder.fit(features)
>>>>>>> Stashed changes
        
        result = np.array(sentiment)
        result = result.reshape(-1,1)
        return result
