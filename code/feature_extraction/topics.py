#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:39:10 2021

@author: tjweber
"""

import numpy as np
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
        
        tweets = inputs[0][:500]
        labels = inputs[1][:500]

        # do tfidf for all tweets instead of only the viral ones, so those that are 
        # common for the viral ones dont lose weight and are filtered out
        vectorizer = TfidfVectorizer(lowercase=False)
        tf_idf_vectors = vectorizer.fit_transform(tweets).todense()
        freq_words = []

        for i in range(0,len(tweets)): 
            vector = tf_idf_vectors[i]
            idx_highest = np.argmax(vector)
            if labels[i] == True:
                freq_words.append(vectorizer.get_feature_names()[idx_highest]) 
            
        counts = Counter(freq_words)
        words = list(counts.keys())
        frequency = list(counts.values())

        # mit komplettem training dataset ausprobieren wie hoch die Schwelle für die 
        # frequency sein sollte
        topics = []
        keywords = []
        for i in range(0, len(words)):
            if frequency[i] >= 3:
                keyword = words[i]
                synsets = wordnet.synsets(keyword)
                topic = [keyword]
                keywords.append(keyword)
                for syn in synsets:
                    synonyms = [str(lemma.name()) for lemma in syn.lemmas()]
                    for synonym in synonyms:
                        if not synonym in topic:
                            topic.append(synonym)
                topics.append(topic)
                
        features = np.full((len(tweets), len(topics)), False, dtype=bool)
        for i in range(0, len(tweets)):
            for j in range(0, len(topics)):
                if any(x in tweets[i] for x in topics[j]):
                    features[i,j] = True

        result = np.array(tweets.str.len())
        result = result.reshape(-1,1)
        return result