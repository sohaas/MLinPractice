#!/bin/bash

# create directory if not yet existing
mkdir -p data/classification/

# run classification on training set (may need to fit classifiers)
echo "  training set"
python -m code.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle --bayes 0.8 0.2 -s 42 --accuracy --kappa --fbeta --sensitivity


# run classification on validation set (with pre-fit classifiers)
echo "  validation set"
python -m code.classification.run_classifier data/dimensionality_reduction/validation.pickle -i data/classification/classifier.pickle --accuracy --kappa --fbeta --sensitivity

# run classification on test set (with pre-fit classifiers) 
echo "  test set"
python -m code.classification.run_classifier data/dimensionality_reduction/validation.pickle -i data/classification/classifier.pickle --accuracy --kappa --fbeta --sensitivity