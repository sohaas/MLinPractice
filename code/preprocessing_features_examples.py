#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some examples for preprocessing and feature extraction.

Created on Wed Oct  6 09:33:29 2021

@author: lbechberger
"""

import string, csv, ast
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# stemming
stemmer = nltk.stem.snowball.SnowballStemmer("english")
print(stemmer.stem("running"))

# bigrams
text = "John Wilkes Booth shot Abraham Lincoln. This did not happen inside the White House."
tokens = nltk.word_tokenize(text)
tokens = [token for token in tokens if token not in string.punctuation]
bigrams = nltk.bigrams(tokens)
freq_dist = nltk.FreqDist(bigrams)
frequency_list = []
for bigram, freq in freq_dist.items():
    frequency_list.append([bigram, freq])
frequency_list.sort(key=lambda x: x[1], reverse=True)
for i in range(10):
    print(frequency_list[i])

# tf-idf
df = pd.read_csv("data/preprocessing/preprocessed.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")
tweets = df["tweet"][:100]

vectorizer = TfidfVectorizer()
tf_idf_vectors = vectorizer.fit_transform(tweets).todense()
print(tf_idf_vectors.shape)
print(vectorizer.get_feature_names()[142:145])
print(tf_idf_vectors[66:71, 142:145])

tf_idf_similarities = cosine_similarity(tf_idf_vectors)
print(tf_idf_similarities[:5, :5])

# obtain tokenized words
tokenized_string = df["tweet_tokenized"][0]
tokenized_list = ast.literal_eval(tokenized_string)
print(type(tokenized_string),type(tokenized_list))
print(tokenized_string[:10], tokenized_list[:5])