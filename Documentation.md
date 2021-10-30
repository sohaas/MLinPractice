# Documentation on Tweet-Prediction Project
This file documents the development process of our tweet prediction tool. 
TODO: Some more introductory stuff

# Goal
Initial goal: "Predict viral tweets"
TODO: Elaborate a bit more

## Evaluation
TODO: Review and improve

For the evaluation of our classifier, we chose several evaluation metrics in 
order to evaluate the performance from different perspectives.
In addition to the accuracy and Cohen's Kappa that were already implemented, 
we decided to add the F Beta Score and the sensitivity. In the following, 
we shortly explain all four of our metrics and motivate the addition of
F Beta and Sensitivity. 

### Design Decisions

Which baselines did you use and why?
TODO: Baselines

#### Accuracy 
Accuracy is the proportion of true results among the total number of cases 
examined. Though widely used as evaluation metric, it is ill-suited for 
unbalanced class distributions as given in our datset. Therefore other 
evaluation metrics like the following are more meaningful in this context. 

#### Cohen's Kappa
Cohen's Kappa adjusts the accuracy for random agreement and is therefore much
more robust against imbalanced class distributions, giving a clearer picture of
the classifier's performance.

#### F Beta Score 
The F Beta Score is a value between 0 and 1, representing a tradeoff between
precision and recall. We decided to use this evaluation metric, as we want our 
model to catch the viral tweets (high recall) without being overly imprecise 
(high precision). The F Beta Score allows for exactly that while also being 
robust against the imbalance of classes in the data.
However by choosing the F Beta score instead of the F1 Score, we have the 
possibility of adjusting the weighting of precision and recall if we find our 
model to miss most of the viral tweets in the attempt of achieving a high 
precision. 

#### Sensitivity 
Sensitivity refers to the true positive rate and summarizes how well the 
positive class has been predicted. Similar to precision in recall, sensitivity 
focuses only on one of the classes (the viral tweets), thereby being robust 
against the imbalanced distribution of viral versus non-viral tweets in the 
dataset. We decided to use this in addition to the fbeta score, as it 
explicitly tells us how well the model did in predicting the positive class, in
our case, the viral tweets. As this matches our goal alomost perfectly, we 
consider this as an important metric to evaluate whether we reached our goal.

### Results

TODO: How do the baselines perform with respect to the evaluation metrics?

### Interpretation

TODO: Is there anything we can learn from these results?

## Preprocessing


### Design Decisions

Before any of the other preprocessing steps, we decided to remove all
unnecessary data to keep it as simple as possible. We did this by first
manually checking for columns with a low rate of entries and then
(computationally) checking for rows with empty tweet entries. All columns and
rows that we found to be not informative (as described) were removed. 

Following that, we removed all links from the original tweets as they were
oftentimes rated highly in the later occuring tf-idf-driven extraction of topics.
As the links are not only present in the tweets, but also stored in a separate 
column of the dataset, the information on urls was not lost, but could still be 
used as a feature. This step was set before the removal of punctuation so that 
the distinctive punctuation of the urls was available to facilitate their 
recognition with a regular expression. After this step, however, punctuation was
removed from the tweets. Similarly, all uppercase letters were lowercased.
We are aware that punctuation and uppercase letters could be used as features
to see if tweets containing, e.g. a lot of exclamation marks or "yelled words"
are likely to become viral. However, we decided to remove them in order to
simplify subsequent content analyses. In case this decision might be revoked
later, a solution might be to simply count their occurences and normalize them
with regard to the length of the tweet.

To set up our more advanced preprocessiong steps (e.g. stemming etc.), 
the tweets were broken down into basic building blocks (first sentences, then 
single words) during tokenization. From that, stopwords (like is, a, they) were
removed to not cludder further processing with nondescript words. At that point,
the tweets were mostly free from interfering non-word information and the 
tokenized form allowed for the continuation of the preprocessing on the word-level.

For our word-level preprocessing, the first idea was to lemmatize the tweet after 
tokenization and stopword removal. We thought lemmatization to be very powerful 
as preparation for our content analysis, as usually during lemmatization the words 
are not merely cut at the root like during stemming ("caring" to "car"), but 
mapped to a an actual meaningful base form ("caring" to "care"). This would enable 
our systems to recognize multiple different forms of a word as semantically equal, 
thereby greatly enhancing the interpretation of the content.
Therefore, we implemented lemmatization, taking into account that it might be 
computationally more expensive. However when taking a look at the data after this 
preprocessing step, we were quite disappointed. Words like "detailed" or "dropping" 
were not transformed to "detail" and "drop" as expected, but stayed the same. 
So long story short, this is why we also implemented a stemmer. This way, we can 
ensure that words with different inflections are cut to the same root form, though
they might be incorrect in meaning or spelling. 

Lastly, all of our previously mentioned preprocessing was brought to use in the 
content analysis of the tweets. There is of course a wide range of approaches,
methods and degrees of detail in which such an analysis can take place. We 
decided to extract common keywords from the viral tweets to get some brought 
content-wise categories as features. Alternatively, methods such as n-grams, 
word-embeddings and much more could be applied here in order to get an analysis
of the content. 
In order to get those content categories, we applied the tf-idf method which is 
able to single out characteristic words of a tweet by uprating all terms that 
are frequent but still unique to a degree. However, even though we were only 
interested in the subset of viral tweets, we still executed the tf-idf scoring 
on all tweets to ensure that very frequent words within specifically the viral 
tweets would not cancel each other out. After we had obtained the tf-idf scores, 
we limited further analysis on the subset of viral tweets. To define a basis for 
our topics, we got the word with the highest tf-idf score from every tweet and 
from those selected the ones that occured more then three times. This resulted
in a considerable amount of frequent words, which we further limited to the 10 
words with the highest frequency. This then, was the starting point for our 
topics. Each of them was extended, by accessing the respective synonyms from
wordnet and adding them to the topic. Initially, we planned on further widening 
and then summarizing them through wordnet hyernyms, but unfortunately, we 
did not have enough time for that. After we had succesfully extracted ten 
supposedly relevant topics from the viral tweets, we compared all tweets with our
topics and stored their occurence. We are aware, that this step could be 
considered as feature extraction instead of preprocessing, but decided to do it
here anyways for simplicity-of-implementation purposes. Because of time issues, 
there are some other shortcomings that we had to accept. One of them is the 
high amount of computational resources that this step takes, partly due to an
inoptimal implementation. Also, initially we planned on using the stemmed tweets 
as input to have a higher comparability of connected wordforms, but unfortunately
did not have enough time to figure out how that would work with wordnet.

TODO: If neccessary, add explanation to tweet limit here

### Results

TODO: Maybe show a short example what your preprocessing does.




## Feature Extraction

### Design DecisionsS

Since it might very well be possible that English tweets are more likely to
become viral, due to the amount of people that speak the language, we
chose this as a feature as well. So, we used the "language" column to
differentiate between English and non-English tweets and test for a correlation
with the virality. The tweets were categorized using boolean values, i.e. 0 =
non-English and 1 = English.

Additionally, we included a feature extractor for checking if the inclusion of
urls in the tweets has any incluence on the virality. For this purpose, the
"urls" column was used to categorize the tweets, again with the help of boolean
values, i.e. 0 = no url included and 1 = url included. We chose this as a
feature, because when looking at the data we noticed that quite often, tweets
contained links as invitations to events, or job oppertunities, or just some
interesting webpage. But the point is that this way, a lot more information can
be transferred than in a tweet without any urls and with a character limit,
which might make the tweet more interesting and therefore more viral.

Beyond the meta data of the tweets, we added the topics extracted beforehand as 
contentwise features, because intuitively it seems obvious that there is some 
kind of relation between the content and the virality of tweets. To not over 
simplify the content, the topics were not summarized into one feature, but each
transformed into an own feature. However as the creation of these topic features
follows the same course of generation, they were created from only one feature
extractor.  

To add to the contentwise features, we decided to categorize the tweets into 
the three categories "positive", "neutral" and "negative" to check for 
correlations regarding the sentiment of a tweet and its virality. To facilitate 
the further handling of these values, we applied an ordinal encoder to map these 
strings to binary numbers. For the categorization, we used the compound score of 
the SentimentIntensityAnalyzer's polarity_scores function. 
According to https://github.com/cjhutto/vaderSentiment#about-the-scoring :
- positive sentiment: compound score >= 0.05
- neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
- negative sentiment: compound score <= -0.05
We decided against using the compound score as it is and categorizing the data
instead to ensure a better comparability.

### Results

TODO: Can you say something about how the feature values are distributed? Maybe show
some plots?

### Interpretation

TODO: Can we already guess which features may be more useful than others?

## Dimensionality Reduction

### Design Decisions

TODO: Which dimensionality reduction technique(s) did you pick and why?

### Results

TODO: Which features were selected / created? Do you have any scores to report?

### Interpretation

TODO: Can we somehow make sense of the dimensionality reduction results?
Which features are the most important ones and why may that be the case?

## Classification

### Design Decisions

### Results

The big finale begins: What are the evaluation results you obtained with your
classifiers in the different setups? Do you overfit or underfit? For the best
selected setup: How well does it generalize to the test set?

### Interpretation

Which hyperparameter settings are how important for the results?
How good are we? Can this be used in practice or are we still too bad?
Anything else we may have learned?
