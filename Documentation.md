# Documentation on Tweet-Prediction Project
This file documents the development process of our tweet prediction tool. 

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
For the evaluation of our classifier, we chose several evaluation metrics in 
order to evaluate the performance from different perspectives.
In addition to the accuracy and Cohen's Kappa that were already implemented, 
we decided to add the F Beta Score and the sensitivity. In the following, 
we shortly explain all four of our metrics and motivate the addition of
F Beta and Sensitivity. 

## Accuracy 
Accuracy is the proportion of true results among the total number of cases 
examined. Though widely used as evaluation metric, it is ill-suited for 
unbalanced class distributions as given in our datset. Therefore other 
evaluation metrics like the following are more meaningful in this context. 

## Cohen's Kappa
Cohen's Kappa adjusts the accuracy for random agreement and is therefore much
more robust against imbalanced class distributions, giving a clearer picture of
the classifier's performance.

## F Beta Score 
The F Beta Score is a value between 0 and 1, representing a tradeoff between
precision and recall. We decided to use this evaluation metric, as we want our 
model to catch the viral tweets (high recall) without being overly imprecise 
(high precision). The F Beta Score allows for exactly that while also being 
robust against the imbalance of classes in the data.
However by choosing the F Beta score instead of the F1 Score, we have the 
possibility of adjusting the weighting of precision and recall if we find our 
model to miss most of the viral tweets in the attempt of achieving a high 
precision. 

## Sensitivity 
Sensitivity refers to the true positive rate and summarizes how well the 
positive class has been predicted. Similar to precision in recall, sensitivity 
focuses only on one of the classes (the viral tweets), thereby being robust 
against the imbalanced distribution of viral versus non-viral tweets in the 
dataset. We decided to use this in addition to the fbeta score, as it 
explicitly tells us how well the model did in predicting the positive class, in
our case, the viral tweets. As this matches our goal alomost perfectly, we 
consider this as an important metric to evaluate whether we reached our goal.