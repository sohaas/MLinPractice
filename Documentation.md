# Documentation on Twitter-Prediction Project
This file documents the development process of our twitter prediction tool. 

# Goal
Initial goal: "Predict viral tweets"

# Preprocessing
For the preprocessing of our data, we decided to convert the tweets to
lowercase in addition to the removal of the punctuation and the tokenization.

We are aware that punctuation and uppercase letters could be used as features
to see if tweets containing, e.g. a lot of exclamation marks or "yelled words"
are likely to become viral.
However, we decided to remove them in order to facilitate subsequent content
analyses. In case this decision might be revoked later, a solution might be
to simply count their occurences and normalize them with regard to the length
of the tweet.

In order to simplify the feature extraction further, the tweets were broken
down into basic building blocks (first into sentences and then single words)
in the tokenization step.

# Feature Extraction

# Dimensionality Reduction

# Machine Learning Model

# Evaluation

