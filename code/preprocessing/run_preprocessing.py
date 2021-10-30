#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps

Created on Tue Sep 28 16:43:18 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from code.preprocessing.links_remover import LinksRemover
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.preprocessing.lowercaser import Lowercaser
from code.preprocessing.tokenizer import Tokenizer
from code.preprocessing.value_handler import ValueHandler
from code.preprocessing.lemmatizer import Lemmatizer
from code.preprocessing.stemmer import Stemmer
from code.preprocessing.stopworder import Stopworder
from code.preprocessing.topic_extractor import TopicExtractor
from code.util import (COLUMN_TWEET, COLUMN_LANGUAGE, COLUMN_TWEET_TOKENS,
                      COLUMN_NO_LINKS, COLUMN_NO_PUNCT, COLUMN_LOWERCASE,
                      COLUMN_NO_STOP, COLUMN_LABEL, SUFFIX_NO_PUNCTUATION,
                      SUFFIX_LOWERCASED, SUFFIX_TOKENIZED, SUFFIX_LEMMATIZED,
                      SUFFIX_STEMMED, SUFFIX_NO_STOPWORDS)

# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-ha", "--handle_values", action = "store_true",
                    help = "handle missing and faulty values")
parser.add_argument("--handle_values_input", help = "input column used for handling values",
                    default = None)
parser.add_argument("-li", "--links", action = "store_true", help = "remove links")
parser.add_argument("--links_input", help = "input column used for links", default = COLUMN_TWEET)
parser.add_argument("-p", "--punctuation", action = "store_true", help = "remove punctuation")
parser.add_argument("--punctuation_input", help = "input column used for punctuation",
                    default = COLUMN_TWEET)
parser.add_argument("-l", "--lowercase", action = "store_true", help = "convert to lowercase")
parser.add_argument("--lowercase_input", help = "input column used for lowercasing",
                    default = COLUMN_NO_PUNCT)
parser.add_argument("-t", "--tokenize", action = "store_true",
                    help = "tokenize given column into individual words")
parser.add_argument("--tokenize_input", help = "input column used for tokenization",
                    default = COLUMN_LOWERCASE)
parser.add_argument("-st", "--stopwords", action = "store_true",
                    help = "remove stopwords of words in given column")
parser.add_argument("--stopwords_input", help = "input column used for removing stopwords",
                    default = COLUMN_TWEET_TOKENS)
parser.add_argument("-s", "--stem", action = "store_true",
                    help = "remove stemming of words in given column")
parser.add_argument("--stem_input", help = "input column used for stemming",
                    default = COLUMN_NO_STOP)
parser.add_argument("-le", "--lemmatize", action = "store_true",
                    help = "lemmatize given column into root words")
parser.add_argument("--lemmatize_input", help = "input column used for lemmatization",
                    default = COLUMN_NO_STOP)
parser.add_argument("-ex", "--extract", action = "store_true",
                    help = "extract topics from given column")
parser.add_argument("--extract_input", help = "input column used for topic extraction",
                    default = COLUMN_NO_STOP)
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location",
                    default = None)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# collect all preprocessors
preprocessors = []
if args.handle_values:
    preprocessors.append(ValueHandler(args.handle_values_input, None))
if args.links:
    preprocessors.append(LinksRemover(args.links_input, COLUMN_NO_LINKS))
if args.punctuation:
    preprocessors.append(PunctuationRemover(args.punctuation_input,
                                            args.punctuation_input + SUFFIX_NO_PUNCTUATION))
# only allow lowercasing to be applied to "_no_punctuation" columns
if args.lowercase and args.lowercase_input.endswith(SUFFIX_NO_PUNCTUATION):
    preprocessors.append(Lowercaser(args.lowercase_input,
                                    args.lowercase_input.partition(SUFFIX_NO_PUNCTUATION)[0]
                                    + SUFFIX_LOWERCASED))
# only allow tokenization of "_lowercased" columns
if args.tokenize and args.tokenize_input.endswith(SUFFIX_LOWERCASED):
    preprocessors.append(Tokenizer(args.tokenize_input,
                                   args.tokenize_input.partition(SUFFIX_LOWERCASED)[0]
                                   + SUFFIX_TOKENIZED))
# only allow stopwords to be removed from "_tokenized" columns
if args.stopwords and args.stopwords_input.endswith(SUFFIX_TOKENIZED):
    preprocessors.append(Stopworder([args.stopwords_input, COLUMN_LANGUAGE],
                                    args.stopwords_input.partition(SUFFIX_TOKENIZED)[0]
                                    + SUFFIX_NO_STOPWORDS))
# only allow lemmatization of "_no_stopwords" columns
if args.lemmatize and args.lemmatize_input.endswith(SUFFIX_NO_STOPWORDS):
    preprocessors.append(Lemmatizer(args.lemmatize_input,
                                    args.tokenize_input.partition(SUFFIX_NO_STOPWORDS)[0]
                                    + SUFFIX_LEMMATIZED))
# only allow stemming of "_no_stopwords" columns
if args.stem and args.stem_input.endswith(SUFFIX_NO_STOPWORDS):
    preprocessors.append(Stemmer([args.stem_input, COLUMN_LANGUAGE],
                                 args.stem_input.partition(SUFFIX_NO_STOPWORDS)[0]
                                 + SUFFIX_STEMMED))
# only allow topic extraction from "_no_stopwords" columns
if args.extract and args.extract_input.endswith(SUFFIX_NO_STOPWORDS):
    preprocessors.append(TopicExtractor([args.extract_input, COLUMN_LABEL], None))

# call all preprocessing steps
for preprocessor in preprocessors:
    df = preprocessor.fit_transform(df)

# store the results
df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

# create a pipeline if necessary and store it as pickle file
if args.export_file is not None:
    pipeline = make_pipeline(*preprocessors)
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)