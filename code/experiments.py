#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:30:07 2021

@author: tjweber
"""

import pandas as pd
import numpy as np
import csv, itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import wordnet

"""# attempting keyword extraction with tf-idf
# do this on stemmed tweets with stopwords and links removed
# maybe take two words from every tweet instead of only one
# easy feature: does one of the keywords appear in the tweet: yes, no; or how
# many of the keywords appeared in the tweet; or a feature for every keyword
# more difficult option: group keywords into topics if possible; extend by 
# synonyms and similar words with the help of wordnet

df = pd.read_csv("data/preprocessing/split/training.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")
df = df[:15000]
tweets = df["tweet"]

# do tfidf for all tweets instead of only the viral ones, so those that are 
# common for the viral ones dont lose weight and are filtered out
vectorizer = TfidfVectorizer()
tf_idf_vectors = vectorizer.fit_transform(tweets).todense()
freq_words = []

for i in range(0,len(tweets)): 
    vector = tf_idf_vectors[i]
    idx_highest = np.argmax(vector)
    if df["label"][i] == True:
        freq_words.append(vectorizer.get_feature_names()[idx_highest]) 
    
counts = Counter(freq_words)
words = list(counts.keys())
frequency = list(counts.values())

# mit komplettem training dataset ausprobieren wie hoch die Schwelle für die 
# frequency sein sollte
keywords = []
for i in range(0, len(words)):
    if frequency[i] >= 3:
        keyword = words[i]
        synsets = wordnet.synsets(keyword)
        topic = [keyword]
        for syn in synsets:
            synonyms = [str(lemma.name()) for lemma in syn.lemmas()]
            for synonym in synonyms:
                if not synonym in topic:
                    topic.append(synonym)
        keywords.append(topic)
        
# print(keywords)

a = np.array([[1,2,3], [4,5,6]])
for column in a.T:
   print(column)"""
   
   
# ordering a dict

exmp_dict = {"hello": 5,
        "my": 2,
        "name": 4,
        "is": 1,
        "Tjorven": 3}

ordered = sorted(exmp_dict.items(), key=lambda x:x[1], reverse=True)

print(ordered)





