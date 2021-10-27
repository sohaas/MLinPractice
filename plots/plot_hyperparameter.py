#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 19:19:38 2021

@author: tjweber
"""

import argparse, csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


METRICS = ["accuracy", "Cohen_kappa", "F-beta score", "sensitivity"]
PARAMS = ["k"]

parser = argparse.ArgumentParser(description = "Hyperparameters of Classifier")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("--plot_name", help = "where to store the generated plots", default = "./plots")
parser.add_argument("--param", type = str, help="parameter to plot", default = None)
parser.add_argument("--metric", type = str, help="metric to be plotted", default = None)
args = parser.parse_args()


def build_df(df, param, metric):
    if (not metric in METRICS) or (not param in PARAMS):   
        print("ERROR: Parameters incorrect")
    else:
        return df[[param, metric, "dataset"]]
    
# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")
df = build_df(df, args.param, args.metric)

# plot parameter against metric
training = df["dataset"] == "training"
training = df["dataset"] == "validation"
line1, = plt.plot(df.iloc[:, 0], df.iloc[:, 1], 'b')
# line2, = plt.plot(n_estimators, test_results, ‘r’, label=”Test AUC”)
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(args.metric)
plt.xlabel(args.param)
plt.show()