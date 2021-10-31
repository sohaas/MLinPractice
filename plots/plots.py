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

def plot_pie(is_viral, labels, title, explode):
    unique, counts = np.unique(is_viral, return_counts = True)
    plt.pie(counts, labels = labels, explode = explode, autopct = "%.1f%%")
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

explode2d = [0.2, 0]
topicLabels = ["Not mentioned", "Mentioned"]


# tfIdf: topic probability
feature = get_features(1)

plot_pie(feature[0], topicLabels, "Topic Feature: 'Probability'", explode2d)
plot_pie(feature[1], topicLabels, "'Probability': Viral Tweets", explode2d)
plot_pie(feature[2], topicLabels, "'Probability': Non-Viral Tweets", explode2d)


# tfIdf: topic picture
feature = get_features(2)

plot_pie(feature[0], topicLabels, "Topic Feature: 'Picture'", explode2d)
plot_pie(feature[1], topicLabels, "'Picture': Viral Tweets", explode2d)
plot_pie(feature[2], topicLabels, "'Picture': Non-Viral Tweets", explode2d)


# tfIdf: topic amp
feature = get_features(3)

plot_pie(feature[0], topicLabels, "Topic Feature: 'AMP'", explode2d)
plot_pie(feature[1], topicLabels, "'AMP': Viral Tweets", explode2d)
plot_pie(feature[2], topicLabels, "'AMP': Non-Viral Tweets", explode2d)


# tfIdf: topic schools
feature = get_features(4)

plot_pie(feature[0], topicLabels, "Topic Feature: 'Schools'", explode2d)
plot_pie(feature[1], topicLabels, "'Schools': Viral Tweets", explode2d)
plot_pie(feature[2], topicLabels, "'Schools': Non-Viral Tweets", explode2d)


# tfIdf: topic vaccine
feature = get_features(5)

plot_pie(feature[0], topicLabels, "Topic Feature: 'Vaccine'", explode2d)
plot_pie(feature[1], topicLabels, "'Vaccine': Viral Tweets", explode2d)
plot_pie(feature[2], topicLabels, "'Vaccine': Non-Viral Tweets", explode2d)


# tfIdf: topic eda (exploratory data analysis)
feature = get_features(6)

plot_pie(feature[0], topicLabels, "Topic Feature: 'EDA'", explode2d)
plot_pie(feature[1], topicLabels, "'EDA': Viral Tweets", explode2d)
plot_pie(feature[2], topicLabels, "'EDA': Non-Viral Tweets", explode2d)


# tfIdf: topic odsc (open data science)
feature = get_features(7)

plot_pie(feature[0], topicLabels, "Topic Feature: 'ODSC'", explode2d)
plot_pie(feature[1], topicLabels, "'ODSC': Viral Tweets", explode2d)
plot_pie(feature[2], topicLabels, "'ODSC': Non-Viral Tweets", explode2d)


# tfIdf: topic graph
feature = get_features(8)

plot_pie(feature[0], topicLabels, "Topic Feature: 'Graph'", explode2d)
plot_pie(feature[1], topicLabels, "'Graph': Viral Tweets", explode2d)
plot_pie(feature[2], topicLabels, "'Graph': Non-Viral Tweets", explode2d)


# tfIdf: topic rstudio
feature = get_features(9)

plot_pie(feature[0], topicLabels, "Topic Feature: 'Rstudio'", explode2d)
plot_pie(feature[1], topicLabels, "'Rstudio': Viral Tweets", explode2d)
plot_pie(feature[2], topicLabels, "'Rstudio': Non-Viral Tweets", explode2d)


# tfIdf: topic cheat
feature = get_features(10)

plot_pie(feature[0], topicLabels, "Topic Feature: 'Cheat'", explode2d)
plot_pie(feature[1], topicLabels, "'Cheat': Viral Tweets", explode2d)
plot_pie(feature[2], topicLabels, "'Cheat': Non-Viral Tweets", explode2d)


# sentiment plots
feature = get_features(11)

plot_pie(feature[0], ["Negative", "Neutral", "Positive"],
         "Sentiment Feature", [0.2, 0, 0])
plot_pie(feature[1], ["Negative", "Neutral", "Positive"],
         "Sentiment Viral Tweets", [0.2, 0, 0])
plot_pie(feature[2], ["Negative", "Neutral", "Positive"],
         "Sentiment Non-Viral Tweets", [0.2, 0, 0])


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
other = count.loc[count.values < 200]
count = count.drop(other.index)
count = count.append(pd.Series({"other": other.sum()}))
plot = count.plot(kind = 'pie', title = "Language Count without EN", figsize = (5,5))
plot.set_ylabel("Language w/o EN")

count = df["language"].groupby(df["language"] == "en").count()
plot = count.plot(kind = 'pie', title = "English vs not English Count", figsize = (5,5))
plot.set_ylabel("Is English")

plot_pie(feature[0], ["Other", "English"], "Language Feature", explode2d)
plot_pie(feature[1], ["Other", "English"], "Language Viral Tweets", explode2d)
plot_pie(feature[2], ["Other", "English"], "Language Non-Viral Tweets", explode2d)


# url  plots
feature = get_features(13)

plot_pie(feature[0], ["W/o URL", "With URL"], "URL Feature", explode2d)
plot_pie(feature[1], ["W/o URL", "With URL"], "URL Viral Tweets", explode2d)
plot_pie(feature[2], ["W/o URL", "With URL"], "URL Non-Viral Tweets", explode2d)