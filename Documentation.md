# Documentation on Twitter-Prediction Project
This file documents the development process of our twitter prediction tool. 

# Goal
Initial goal: "Predict viral tweets"

# Preprocessing
For the preprocessing of our data, we decided to convert the tweets to lowercase in addition to the removal of the punctuation and the tokenization.

We are aware that punctuation and uppercase letters could be used as features to see if tweets containing, e.g. a lot of exclamation marks or "yelled words" are likely to become viral.
However, we decided to remove them in order to facilitate subsequent content analyses.
In case this decision might be revoked later, a solution might be to simply count their occurences and normalize them with regard to the length of the tweet.

In order to enable further preprocessiong steps, like lemmatization, the tweets were broken down into basic building blocks, i.e. single words by tokenization.

First, our idea was to lemmatize the tweet after tokenizing it to optimize the data for the subsequent content analysis.
The reason why we initially chose the approach of lemmatization over stemming was that in stemming, words are merely cut and not mapped to a meaningful base form like in lemmatization. 
Therefore, we assumed that, e.g. the word "caring" would become "car" with stemming, whereas lemmatization would correctly transform it to "care". Therefore, we implemented lemmatization, taking into account that it might be computationally more expensive.
However when taking a look at the data after this preprocessing step, we were quite disappointed. Words like "detailed" or "dropping" were not transformed to "detail" and "drop" as expected, but stayed the same.
So long story short, this is why we also implemented a stemmer. This way, we can ensure that words with different inflections are cut to the same root form, though they might be incorrect in meaning or spelling. 

# Feature Extraction

# Dimensionality Reduction

# Machine Learning Model

# Evaluation

