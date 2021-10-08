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

In order to enable further preprocessiong steps, like lemmatization, the tweets
were broken down into basic building blocks, i.e. single words by tokenization.

After tokenizing the tweet, it can be lemmatized, i.e. the inflections can be
removed and the words mapped to their root from. This step serves to optimize
the data for the subsequent content analysis. An alternative would have been to
implement stemming instead of lemmatization, however in stemming, words are
merely cut and not mapped to their root form. Therefore, e.g. the word "caring"
becomes "car" with stemming, whereas lemmatization correctly transforms it to
"care", which is why we decided for lemmatization, though this might be
computationally more expensive.

# Feature Extraction

# Dimensionality Reduction

# Machine Learning Model

# Evaluation

