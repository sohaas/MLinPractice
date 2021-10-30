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
from matplotlib import pyplot as plt

df = pd.read_csv("data/preprocessing/preprocessed.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

with open("data/feature_extraction/training.pickle", "rb") as f_in:
    data = pickle.load(f_in)

features = data["features"]
labels = np.squeeze(data["labels"])


# character length plots
feature = features[:, 0]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]

plt.hist(feature, range = [0,400])
plt.title("Character Count Feature")
plt.xlabel("Characters")
plt.ylabel("Tweets")

plt.hist([pos, neg], range = [0,400], alpha=0.5, label = ["Viral", "Non-Viral"])
plt.legend(loc='upper right')
plt.title("Characters Viral vs Non-Viral Tweets")
plt.xlabel("Characters")
plt.ylabel("Tweets")


# tfIdf: topic probability
feature = features[:, 1]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]


# tfIdf: topic picture
feature = features[:, 2]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]


# tfIdf: topic amp
feature = features[:, 3]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]


# tfIdf: topic schools
feature = features[:, 4]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]


# tfIdf: topic vaccine
feature = features[:, 5]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]


# tfIdf: topic eda
feature = features[:, 6]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]


# tfIdf: topic odsc
feature = features[:, 7]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]


# tfIdf: topic graph
feature = features[:, 8]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]


# tfIdf: topic rstudio
feature = features[:, 9]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]


# tfIdf: topic cheat
feature = features[:, 10]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]


# sentiment plots
feature = features[:, 11]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]


# language plots
feature = features[:, 12]
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

unique, counts = np.unique(feature, return_counts = True)
plt.pie(counts, labels = ["Not English", "English"], explode = [0.2, 0], autopct = "%.1f%%")
plt.title("Language Feature")
plt.show()

unique, counts = np.unique(pos, return_counts = True)
plt.pie(counts, labels = ["Not English", "English"], explode = [0.2, 0], autopct = "%.1f%%")
plt.title("Language Viral Tweets")
plt.show()

unique, counts = np.unique(neg, return_counts = True)
plt.pie(counts, labels = ["Not English", "English"], explode = [0.2, 0], autopct = "%.1f%%")
plt.title("Language Non-Viral Tweets")
plt.show()


# url  plots
feature = features[:, 13]
# viral tweets
pos = feature[labels]
# non-viral tweets
neg_index = np.array([not x for x in labels])
neg = feature[neg_index]