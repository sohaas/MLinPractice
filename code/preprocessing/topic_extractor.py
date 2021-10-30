#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 18:26:20 2021

@author: tjweber
"""

from code.preprocessing.preprocessor import Preprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import wordnet
import ast

# extracts topics from preprocessed viral tweets
class TopicExtractor(Preprocessor):
    
    # constructor
    def __init__(self, input_column, output_column):
        # input column "tweet", new output column
        super().__init__([input_column[0], input_column[1]], output_column)
    
    # don't need to implement _set_variables()
    def _set_variables(self, inputs):
        self.tweets = inputs[0]
        self.labels = inputs[1]
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs, df):
        print("Extracting topics")
        print("    Warning: For runtime reasons, this step is only performed ",
              "on 50.000 of the tweets. If you have 4-6h extra time and a good ",
              "computer, feel free to remove the limitation in line 65 of ",
              "code/preprocessing/topic_extractor.py")
          
        # get words with highest tf_idf score
        freq_words = self._get_freq_words()
                
        # get topics (= keywords and their synonyms)
        counted = Counter(freq_words)
        ordered = sorted(counted.items(), key=lambda x:x[1], reverse=True)
        if len(ordered) > 10:
            ordered = ordered[:10]
        topics = self._get_topics(ordered)
             
        # search each tweet for topics and store
        features = np.full((len(self.tweets), len(topics)), False, dtype=bool)
        for i in range(0, len(self.tweets)):
            tweet_list = ast.literal_eval(self.tweets[i])
            for j in range(0, len(topics)):
                if (set(tweet_list) & set(topics[j])):
                    features[i,j] = True
         
        # return list of features
        features_list = []
        for column in features.T:
            features_list.append(pd.DataFrame(data=column))       
            
        return features_list         
    
    # get frequent words with high tf-idf score    
    def _get_freq_words(self):
        limit = 50000
        vectorizer = TfidfVectorizer(lowercase=False)
        tf_idf_vectors = vectorizer.fit_transform(self.tweets[:limit]).todense()
        
        # add index column to dataframe
        frame = { 'tweets': self.tweets[:limit], 'labels': self.labels[:limit] } 
        df_new = pd.DataFrame(frame)
        df_new["index"] = range(0, len(self.tweets[:limit]))
         
        # get entries labeled as True
        df_new = df_new.loc[df_new['labels'] == True]
         
        # store words with highest tf_idf scores 
        freq_words = []
        for i in range(0, df_new.shape[0]):
            idx_highest = np.argmax(tf_idf_vectors[df_new.index[i]])
            freq_words.append(vectorizer.get_feature_names()[idx_highest])
            if i % 100 == 0:
                progress = round((i * 100 / df_new.shape[0]), 3)
                print("    Progress: {0}%".format(progress), end = "\r")
        print("\n")
        return freq_words
    
    
    # get synonyms of word from wordnet 
    def _get_synonyms(self, word):
        synsets = wordnet.synsets(word)
        synonyms = []
        for syn in synsets:
            synonyms += [str(lemma.name()) for lemma in syn.lemmas()]     
        return synonyms
    
    # get popular topics based on keywords and their frequency
    def _get_topics(self, word_freqs):
        topics = []
        self._output_column = []
        for word, freq in word_freqs:
            if freq >= 3:
                topic = [word] + self._get_synonyms(word)
                topics.append(list(set(topic)))
                self._output_column.append("topic_" + word)
        return topics
        
