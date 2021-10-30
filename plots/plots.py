#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 22:12:09 2021

@author: sohaas
"""

import csv
import pickle
import pandas as pd
import numpy as np

df = pd.read_csv("data/preprocessing/preprocessed.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

with open("data/feature_extraction/training.pickle", "rb") as f_in:
    data = pickle.load(f_in)

features = data["features"]
labels = data["labels"]

# feature plots

# character length plots
feature = features[:, 0].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# tfidf plots

# tfIdf: topic probability
feature = features[:, 1].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# tfIdf: topic picture
feature = features[:, 2].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# tfIdf: topic amp
feature = features[:, 3].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# tfIdf: topic schools
feature = features[:, 4].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# tfIdf: topic vaccine
feature = features[:, 5].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# tfIdf: topic eda
feature = features[:, 6].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# tfIdf: topic odsc
feature = features[:, 7].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# tfIdf: topic graph
feature = features[:, 8].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# tfIdf: topic rstudio
feature = features[:, 9].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# tfIdf: topic cheat
feature = features[:, 10].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# sentiment plots
feature = features[:, 11].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

# language plots
feature = features[:, 12].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

count = df["language"].value_counts()
plot = count.plot(kind = "bar", logy = True, title = "Language Count", figsize = (15,5))
plot.set_xlabel("Language")
plot.set_ylabel("Count")

count = df.loc[df["language"]!="en", "language"].value_counts()
plot = count.plot(kind = 'bar', title = "Language Count without EN", figsize = (15,5))
plot.set_xlabel("Language w/o EN")
plot.set_ylabel("Count")

count = df.loc[df["language"]!="en", "language"].value_counts()
plot = count.plot(kind = 'pie', title = "Language Count without EN", figsize = (5,5))
plot.set_ylabel("Language w/o EN")

count = df["language"].groupby(df["language"] == "en").count()
plot = count.plot(kind = 'pie', title = "English vs not English Count", figsize = (5,5))
plot.set_ylabel("Is English")

# url  plots
feature = features[:, 13].reshape(features.shape[0], 1)
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]