#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of feature extractors.

Created on Wed Sep 29 11:00:24 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
import numpy as np
from code.feature_extraction.character_length import CharacterLength
from code.feature_extraction.topics import Topics
from code.feature_extraction.sentiment_analyzer import SentimentAnalyzer
from code.feature_extraction.language_en import EnglishLanguage
from code.feature_extraction.url_included import UrlIncluded
from code.feature_extraction.feature_collector import FeatureCollector
from code.util import COLUMN_TWEET, COLUMN_LABEL, COLUMN_LANGUAGE, COLUMN_URL

# setting up CLI
parser = argparse.ArgumentParser(description = "Feature Extraction")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file",
                    help = "create a pipeline and export to the given location",
                    default = None)
parser.add_argument("-i", "--import_file",
                    help = "import an existing pipeline from the given location",
                    default = None)
parser.add_argument("-c", "--char_length", action = "store_true",
                    help = "compute the number of characters in the tweet")
parser.add_argument("-t", "--topics", action = "store_true",
                    help = "access topics present in the tweet")
parser.add_argument("-s", "--sentiment", action = "store_true",
                    help = "analyze the sentiment of the tweet")
parser.add_argument("-l", "--language", action = "store_true",
                    help = "analyze whether the tweet is in English")
parser.add_argument("-u", "--url", action = "store_true",
                    help = "analyze whether the tweet contains an URL")
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

if args.import_file is not None:
    # simply import an exisiting FeatureCollector
    with open(args.import_file, "rb") as f_in:
        feature_collector = pickle.load(f_in)

# need to create FeatureCollector manually
else:

    # collect all feature extractors
    features = []
    # character length of original tweet (without any changes)
    if args.char_length:
        features.append(CharacterLength(COLUMN_TWEET))
    # topics of preprocessed tweet
    if args.topics:
        input_cols = list(filter(lambda x: "topic_" in x, list(df.columns)))
        features.append(Topics(input_cols))
    # sentiment of original tweet (without any changes)
    if args.sentiment:
        features.append(SentimentAnalyzer(COLUMN_TWEET))
    # language of original tweet (without any changes)
    if args.language:
        features.append(EnglishLanguage(COLUMN_LANGUAGE))
    # urls of original tweet (without any changes)
    if args.url:
        features.append(UrlIncluded(COLUMN_URL))
    
    # create overall FeatureCollector
    feature_collector = FeatureCollector(features)
    # fit it on the given data set (assumed to be training data)
    feature_collector.fit(df)


# apply the given FeatureCollector on the current data set
# maps the pandas DataFrame to an numpy array
feature_array = feature_collector.transform(df)

# get label array
label_array = np.array(df[COLUMN_LABEL])
label_array = label_array.reshape(-1, 1)

# store the results
results = {"features": feature_array, "labels": label_array, 
           "feature_names": feature_collector.get_feature_names()}
with open(args.output_file, 'wb') as f_out:
    pickle.dump(results, f_out)

# export the FeatureCollector as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(feature_collector, f_out)