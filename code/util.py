#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""

# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"
COLUMN_LANGUAGE = "language"
COLUMN_URL = "urls"

# columns to be removed
COLUMNS_REMOVE = ["place", "cashtags", "retweet", "near", "geo", "source",
                  "user_rt_id", "user_rt", "retweet_id", "retweet_date", 
                  "translate", "trans_src", "trans_dest"]

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_NO_PUNCT = "tweet_no_punctuation"
COLUMN_LOWERCASE = "tweet_lowercased"
COLUMN_TWEET_TOKENS = "tweet_tokenized"
COLUMN_NO_STOP = "tweet_no_stopwords"
COLUMN_NO_LINKS = "tweet_no_links"

# suffixes for new columns
SUFFIX_NO_PUNCTUATION = "_no_punctuation"
SUFFIX_LOWERCASED = "_lowercased"
SUFFIX_TOKENIZED = "_tokenized"
SUFFIX_LEMMATIZED = "_lemmatized"
SUFFIX_STEMMED = "_stemmed"
SUFFIX_NO_STOPWORDS = "_no_stopwords"