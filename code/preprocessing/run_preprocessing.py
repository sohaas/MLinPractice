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
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.preprocessing.lowercaser import Lowercaser
from code.preprocessing.tokenizer import Tokenizer
from code.preprocessing.lemmatizer import Lemmatizer
from code.preprocessing.stemmer import Stemmer
from code.util import COLUMN_TWEET, COLUMN_TWEET_TOKENS, SUFFIX_NO_PUNCTUATION, SUFFIX_LOWERCASED, SUFFIX_TOKENIZED, SUFFIX_LEMMATIZED, SUFFIX_STEMMED

# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-p", "--punctuation", action = "store_true", help = "remove punctuation")
parser.add_argument("--punctuation_input", help = "input column used for punctuation", default = COLUMN_TWEET)
parser.add_argument("-l", "--lowercase", action = "store_true", help = "convert to lowercase")
parser.add_argument("--lowercase_input", help = "input column used for lowercasing", default = COLUMN_TWEET)
parser.add_argument("-t", "--tokenize", action = "store_true", help = "tokenize given column into individual words")
parser.add_argument("--tokenize_input", help = "input column used for tokenization", default = COLUMN_TWEET)
parser.add_argument("-le", "--lemmatize", action = "store_true", help = "lemmatize given column into root words")
parser.add_argument("--lemmatize_input", help = "input column used for lemmatization", default = COLUMN_TWEET_TOKENS)
parser.add_argument("-s", "--stem", action = "store_true", help = "remove stemming of words in given column")
parser.add_argument("--stem_input", help = "input column used for stemming", default = COLUMN_TWEET_TOKENS)
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# collect all preprocessors
preprocessors = []
if args.punctuation:
    preprocessors.append(PunctuationRemover(args.punctuation_input, args.punctuation_input + SUFFIX_NO_PUNCTUATION))
if args.lowercase:
    preprocessors.append(Lowercaser(args.lowercase_input, args.lowercase_input + SUFFIX_LOWERCASED))
if args.tokenize:
    preprocessors.append(Tokenizer(args.tokenize_input, args.tokenize_input + SUFFIX_TOKENIZED))
# only allow "_tokenized" colums to be lemmatized
if args.lemmatize and args.lemmatize_input.endswith(SUFFIX_TOKENIZED):
        preprocessors.append(Lemmatizer(args.lemmatize_input, args.lemmatize_input.partition(SUFFIX_TOKENIZED)[0] + SUFFIX_LEMMATIZED))
# only allow "_tokenized" colums to be stemmed
if args.stem and args.stem_input.endswith(SUFFIX_TOKENIZED):
        preprocessors.append(Stemmer(args.stem_input, args.stem_input.partition(SUFFIX_TOKENIZED)[0] + SUFFIX_STEMMED))

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