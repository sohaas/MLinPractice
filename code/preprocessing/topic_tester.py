#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 23:01:23 2021

@author: ml
"""

import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/preprocessing/labeled.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")
tweets = df["tweet"][:10000]

vectorizer = TfidfVectorizer(lowercase=False)
tf_idf_vectors = vectorizer.fit_transform(tweets).todense()

# add index column to dataframe
frame = { 'tweets': tweets, 'labels': df["label"][:10000] } 
df_new = pd.DataFrame(frame)
df_new["index"] = range(0, len(tweets))
 
# get entries labeled as True
df_new = df_new.loc[df_new['labels'] == True]
 
# store words with highest tf_idf scores 
freq_words = []
for i in range(0, df_new.shape[0]):
    idx_highest = np.argmax(tf_idf_vectors[df_new.index[i]])
    freq_words.append(vectorizer.get_feature_names()[idx_highest])
