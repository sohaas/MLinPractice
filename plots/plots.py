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

df = pd.read_csv("data/preprocessing/preprocessed.csv",
                 quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")


with open("data/feature_extraction/training.pickle", "rb") as f_in:
    data = pickle.load(f_in)

    features = data["features"]
    labels = np.squeeze(data["labels"])

def get_features(column):
    feature = features[:, column]
    # viral tweets
    pos = feature[labels]
    # non-viral tweets
    neg_index = np.array([not x for x in labels])
    neg = feature[neg_index]
    return [feature, pos, neg]

def plot_pie(is_viral, labels, title):
    unique, counts = np.unique(is_viral, return_counts = True)
    plt.pie(counts, labels = labels, explode = [0.2, 0], autopct = "%.1f%%")
    plt.title(title)
    plt.show()
    return


# character length plots
feature = get_features(0)

plt.hist(feature[0], range = [0,400])
plt.title("Character Count Feature")
plt.xlabel("Characters")
plt.ylabel("Tweets")

plt.hist([feature[1], feature[2]], range = [0,400], label = ["Viral", "Non-Viral"])
plt.legend(loc='upper right')
plt.title("Characters Viral vs Non-Viral Tweets")
plt.xlabel("Characters")
plt.ylabel("Tweets")


# tfIdf: topic probability
feature = get_features(1)

plot_pie(feature[0], ["Not mentioned", "Mentioned"], "Topic Feature: 'Probability'")
plot_pie(feature[1], ["Not mentioned", "Mentioned"], "'Probability': Viral Tweets")
plot_pie(feature[2], ["Not mentioned", "Mentioned"], "'Probability': Non-Viral Tweets")


# tfIdf: topic picture
feature = get_features(2)


# tfIdf: topic amp
feature = get_features(3)


# tfIdf: topic schools
feature = get_features(4)


# tfIdf: topic vaccine
feature = get_features(5)


# tfIdf: topic eda
feature = get_features(6)


# tfIdf: topic odsc
feature = get_features(7)


# tfIdf: topic graph
feature = get_features(8)


# tfIdf: topic rstudio
feature = get_features(9)


# tfIdf: topic cheat
feature = get_features(10)


# sentiment plots
feature = get_features(11)


# language plots
feature = get_features(12)

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

unique, counts = np.unique(feature[0], return_counts = True)
plt.pie(counts, labels = ["Not English", "English"], explode = [0.2, 0], autopct = "%.1f%%")
plt.title("Language Feature")
plt.show()

unique, counts = np.unique(feature[1], return_counts = True)
plt.pie(counts, labels = ["Not English", "English"], explode = [0.2, 0], autopct = "%.1f%%")
plt.title("Language Viral Tweets")
plt.show()

unique, counts = np.unique(feature[2], return_counts = True)
plt.pie(counts, labels = ["Not English", "English"], explode = [0.2, 0], autopct = "%.1f%%")
plt.title("Language Non-Viral Tweets")
plt.show()


# url  plots
feature = get_features(13)