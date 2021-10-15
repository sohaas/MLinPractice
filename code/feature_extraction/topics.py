#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:39:10 2021

@author: tjweber
"""

import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import wordnet
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class Topics(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column[0], input_column[1]], "{0}_topics".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the word length based on the inputs
    def _get_values(self, inputs):
        
        tweets_lim = inputs[0][:10]
        tweets = inputs[0]
        labels = inputs[1]

        # get tf_idf vectors
        vectorizer = TfidfVectorizer(lowercase=False)
        tf_idf_vectors = vectorizer.fit_transform(tweets_lim).todense()

        # store words with highest scores
        freq_words = []
        for i in range(0,len(tweets_lim)): 
            if labels[i] == True:
                idx_highest = np.argmax(tf_idf_vectors[i])
                freq_words.append(vectorizer.get_feature_names()[idx_highest]) 
           
        # get keywords and their synonyms
        counts = Counter(freq_words)
        words = list(counts.keys())
        frequency = list(counts.values())
        topics = []
        for i in range(0, len(words)):
            if frequency[i] >= 3:
                topic = [words[i]]
                synsets = wordnet.synsets(words[i])
                for syn in synsets:
                    synonyms = [str(lemma.name()) for lemma in syn.lemmas()]
                    for synonym in synonyms:
                        if not synonym in topic:
                            topic.append(synonym)
                topics.append(topic)
                
        features = np.zeros((len(tweets), len(topics)))
        for i in range(0, len(tweets)):
            tweet = tweets[i]
            tweet_list = ast.literal_eval(tweets[i])
            for j in range(0, len(topics)):
                if any(x in tweet for x in topics[j]):
                    features[i,j] = 1
                    # print("topics: ", topics[j])
                    # print("tweet: ", tweets[i])
        
        features_list = []
        for column in features.T:
            features_list.append(column.reshape(-1,1))
            
        return features_list
