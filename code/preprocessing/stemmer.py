#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove stems from the tokenized words of the tweet.
Attention: Must only be applied on a "_tokenized" column!

Created on Fri Oct  8 10:09:23 2021

@author: sohaas
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.stem import SnowballStemmer
import ast

class Stemmer(Preprocessor):
    """Removes the stems from the words of the given input column."""
    
    def __init__(self, input_column, output_column):
        """Initialize the Stemmer with the given input and output column."""
        super().__init__(input_column, output_column)
    
    # don't need to implement _set_variables(), since no variables to set
    
    def _get_values(self, inputs):
        """Stem the tweet."""
        
        stemmed = []
        
        for index, value in inputs[0].items():
            tokenized_list = ast.literal_eval(value)
            stemmed_tweet = []
            
            stemmer = self.get_stemmer(inputs[1][index])
            
            for word in tokenized_list:
                if stemmer is not None:
                    stemmed_word = stemmer.stem(word)
                    stemmed_tweet.append(stemmed_word)
                else:
                    stemmed_tweet.append(word)
                
            stemmed.append(str(stemmed_tweet))
        
        return stemmed

    def get_stemmer(self, language):
        """Return list of stopwords in the right language."""
        
        if language == "ar":
            return SnowballStemmer("arabic")
        elif language == "da":
            return SnowballStemmer("danish")
        elif language == "nl":
            return SnowballStemmer("dutch")
        elif language == "en":
            return SnowballStemmer("english")
        elif language == "fi":
            return SnowballStemmer("finnish")
        elif language == "fr":
            return SnowballStemmer("french")
        elif language == "de":
            return SnowballStemmer("german")
        elif language == "hu":
            return SnowballStemmer("hungarian")
        elif language == "it":
            return SnowballStemmer("italian")
        elif language == "no":
            return SnowballStemmer("norwegian")
        elif language == "pt":
            return SnowballStemmer("portuguese")
        elif language == "ro":
            return SnowballStemmer("romanian")
        elif language == "ru":
            return SnowballStemmer("russian")
        elif language == "es":
            return SnowballStemmer("spanish")
        elif language == "sv":
            return SnowballStemmer("swedish")
        else:
            return None
    
    
# -*- coding: utf-8 -*-

