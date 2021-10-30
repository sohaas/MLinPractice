#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a dimensionality reduction technique.

Created on Wed Sep 29 13:33:37 2021

@author: lbechberger
"""

import argparse, pickle
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# setting up CLI
parser = argparse.ArgumentParser(description = "Dimensionality reduction")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file",
                    help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file",
                    help = "import an existing pipeline from the given location", default = None)
parser.add_argument("-r", "--recursive", type = int,
                    help = "select n best features with recursive frature elimination using a logistic regression model",
                    default = None)
parser.add_argument("-s", "--select_from_model", type = int,
                    help = "select maximally n features from random forest classifier", default = None)
parser.add_argument("-m", "--mutual_information", type = int,
                    help = "select K best features with Mutual Information", default = None)
parser.add_argument("--verbose", action = "store_true",
                    help = "print information about feature selection process")
args = parser.parse_args()

# load the data
with open(args.input_file, 'rb') as f_in:
    input_data = pickle.load(f_in)

features = input_data["features"]
labels = input_data["labels"]
feature_names = input_data["feature_names"]

if args.import_file is not None:
    # simply import an already fitted dimensionality reducer
    with open(args.import_file, 'rb') as f_in:
        dim_red = pickle.load(f_in)

# need to set things up manually
else:

    # select K best based on Mutual Information
    if args.mutual_information is not None:
        dim_red = SelectKBest(mutual_info_classif, k = args.mutual_information)
        dim_red.fit(features, labels.ravel())
        
        # resulting feature names based on support given by SelectKBest
        def get_feature_names(kbest, names):
            support = kbest.get_support()
            result = []
            for name, selected in zip(names, support):
                if selected:
                    result.append(name)
            return result
        
        if args.verbose:
            print("    SelectKBest with Mutual Information and k = {0}".format(args.mutual_information))
            print("    {0}".format(feature_names))
            print("    " + str(dim_red.scores_))
            print("    " + str(get_feature_names(dim_red, feature_names)))

    # select n best features based on RFE/LogReg
    elif args.recursive is not None:
        estimator = LogisticRegression(random_state = 42, max_iter = 10000)
        dim_red = RFE(estimator, n_features_to_select = args.recursive)
        dim_red.fit(features, labels.ravel())
        
        # resulting feature names based on support given by RFE
        def get_feature_names(recursive, names):
            support = recursive.get_support()
            result = []
            for name, selected in zip(names, support):
                if selected:
                    result.append(name)
            return result
        
        if args.verbose:
            print("    RFE with Logistic Regression and n = {0}".format(args.recursive))
            print("    {0}".format(feature_names))
            print("    " + str(dim_red.ranking_))
            print("    " + str(get_feature_names(dim_red, feature_names)))
    
    # select n best features from random forest classifier
    elif args.select_from_model is not None:
        estimator = RandomForestClassifier(n_estimators = 10, random_state=42)
        estimator.fit(features, labels.ravel())
        dim_red = SelectFromModel(estimator, threshold = 0.1, prefit = True, max_features = args.select_from_model)
        
        # resulting feature names based on support given by SelectFromModel
        def get_feature_names(sfm, names):
            support = sfm.get_support()
            result = []
            for name, selected in zip(names, support):
                if selected:
                    result.append(name)
            return result
        
        if args.verbose:
            print("    Select from model with Random Forest Classifier and n = {0}".format(args.select_from_model))
            print("    {0}".format(feature_names))
            print("    " + str(estimator.feature_importances_))
            print("    " + str(get_feature_names(dim_red, feature_names)))
        

# apply the dimensionality reduction to the given features
reduced_features = dim_red.transform(features)

# store the results
output_data = {"features": reduced_features, "labels": labels}
with open(args.output_file, 'wb') as f_out:
    pickle.dump(output_data, f_out)

# export the dimensionality reduction technique as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(dim_red, f_out)