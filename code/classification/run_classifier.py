#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, fbeta_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from mlflow import log_metric, log_param, set_tracking_uri, start_run, end_run

# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int,
                    help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file",
                    help = "export the trained classifier to the given location",
                    default = None)
parser.add_argument("-i", "--import_file",
                    help = "import a trained classifier from the given location",
                    default = None)
parser.add_argument("-m", "--majority", action = "store_true",
                    help = "majority class classifier")
parser.add_argument("-f", "--frequency", action = "store_true",
                    help = "label frequency classifier")
parser.add_argument("-b", "--bayes", nargs = "*", type = float, 
                    help = "gaussian naive bayes classifier")
parser.add_argument("--knn", type = int,
                    help = "k nearest neighbor classifier with the specified value of k",
                    default = None)
parser.add_argument("--rf", type = int, help = "random forest classifier",
                    default = None)
parser.add_argument("--rf_cw", type = str, help = "random forest classifier",
                    default = None)
parser.add_argument("--svm", nargs = "*", type = int,
                    help = "support vector machine classifier", default = None)
parser.add_argument("--kernel", type = str,
                    help = "support vector machine classifier", default = None)
parser.add_argument("-a", "--accuracy", action = "store_true",
                    help = "evaluate using accuracy")
parser.add_argument("-k", "--kappa", action = "store_true",
                    help = "evaluate using Cohen's kappa")
parser.add_argument("-fb", "--fbeta", action = "store_true",
                    help = "evaluate using F-beta score")
parser.add_argument("-se", "--sensitivity", action = "store_true",
                    help = "evaluate using sensitivity")
parser.add_argument("--log_folder", help = "where to log the mlflow results",
                    default = "data/classification/mlflow")
args = parser.parse_args()

# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)


set_tracking_uri(args.log_folder)
start_run()

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        input_dict = pickle.load(f_in)         
    
    classifier = input_dict["classifier"]
    for param, value in input_dict["params"].items():
        log_param(param, value)
    
    log_param("dataset", "validation")

# manually set up a classifier
else:
    
    # majority vote classifier
    if args.majority:
        print("    majority vote classifier")
        log_param("classifier", "majority")
        params = {"classifier": "majority"}
        classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)
    # label frequency classifier
    elif args.frequency:
        print("    label frequency classifier")
        log_param("classifier", "frequency")
        params = {"classifier": "frequency"}
        classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)
    # categorical naive bayes classifier        
    elif args.bayes is not None:
        print("    gaussian naive bayes classifier")
        print(args.bayes)
        log_param("classifier", "bayes")
        log_param("priors", args.bayes)
        params = {"classifier": "bayes", "priors": args.bayes}
        standardizer = StandardScaler()
        bayes_classifier = CategoricalNB(class_prior=args.bayes)
        classifier = make_pipeline(standardizer, bayes_classifier)
    # k-nearest neighbor classifier
    elif args.knn is not None:
        print("    {0} nearest neighbor classifier".format(args.knn))
        log_param("classifier", "knn")
        log_param("k", args.knn)
        params = {"classifier": "knn", "k": args.knn}
        standardizer = StandardScaler()
        knn_classifier = KNeighborsClassifier(n_neighbors=args.knn, n_jobs = -1)
        classifier = make_pipeline(standardizer, knn_classifier)
    # random forest classifier
    elif args.rf is not None:
        print("    random forest classifier with {0} trees".format(args.rf))
        log_param("classifier", "rf")
        log_param("trees", args.rf)
        log_param("class_weight", args.rf_cw)
        params = {"classifier": "rf", "trees": args.rf, "class_weight": args.rf_cw}
        standardizer = StandardScaler()
        rf_classifier = RandomForestClassifier(n_estimators=args.rf, class_weight=args.rf_cw, n_jobs = -1)
        classifier = make_pipeline(standardizer, rf_classifier)
    # support vector machine classifier
    elif args.kernel == "linear" or args.svm == "polynomial" or args.svm == "rbf" or args.svm == "sigmoid":
        print("    support vector machine classifier with {0} kernel".format(args.svm))
        log_param("classifier", "svm")
        log_param("kernel", args.kernel)
        log_param("class weights", args.svm)
        params = {"classifier": "svm", "kernel": args.kernel, "c weights": args.svm}
        standardizer = StandardScaler()
        svm_classifier = svm.SVC(kernel=args.kernel, class_weight={0:args.svm[0], 1:args.svm[1]})
        classifier = make_pipeline(standardizer, svm_classifier)
    
    classifier.fit(data["features"], data["labels"].ravel())
    log_param("dataset", "training")

# now classify the given data
prediction = classifier.predict(data["features"])

# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("accuracy", accuracy_score))
if args.kappa:
    evaluation_metrics.append(("Cohen_kappa", cohen_kappa_score))
if args.fbeta:
    evaluation_metrics.append(("F-beta score", fbeta_score))
if args.sensitivity:
    target_names = ["non-viral", "viral"]
    cl_report = classification_report(data["labels"], prediction,
                                      target_names=target_names, zero_division=0, output_dict=True)
    evaluation_metrics.append(("sensitivity", cl_report["viral"]["recall"]))

# compute and print them
for metric_name, metric in evaluation_metrics:
    if metric_name == "F-beta score":
        metric_value = metric(data["labels"], prediction, beta=1.2)
    elif metric_name == "sensitivity":
        metric_value = metric
    else:
        metric_value = metric(data["labels"], prediction)
    print("    {0}: {1}".format(metric_name, metric_value))
    log_metric(metric_name, metric_value)
    
# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    output_dict = {"classifier": classifier, "params": params}
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(output_dict, f_out)
        
end_run()