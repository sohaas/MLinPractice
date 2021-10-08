#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove stopwords from the tokenized words of the tweet.
Attention: Must only be applied on a "_tokenized" column!

Created on Fri Oct  8 12:17:46 2021

@author: sohaas
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.corpus import stopwords
import ast

class Stopworder(Preprocessor):
    """Removes the stopwords from the words of the given input column."""
    
    def __init__(self, input_column, output_column):
        """Initialize the Stopworder with the given input and output column."""
        super().__init__(input_column, output_column)
    
    # don't need to implement _set_variables(), since no variables to set
    
    def _get_values(self, inputs):
        """Remove stopwords."""
        
        no_stopwords = []
        
        for index, value in inputs[0].items():
            tokenized_list = ast.literal_eval(value)
            no_stopwords_tweet = []
            
            stopword_list = self.get_stopword_list(inputs[1][index])
                
            for word in tokenized_list:
                if word.lower() not in stopword_list:
                    no_stopwords_tweet.append(word)
                
            no_stopwords.append(str(no_stopwords_tweet))
        
        return no_stopwords

    def get_stopword_list(self, language):
        """Return list of stopwords in the right language."""
        
        if language == "ar":
            return set(stopwords.words("arabic"))
        elif language == "da":
            return set(stopwords.words("danish"))
        elif language == "nl":
            return set(stopwords.words("dutch"))
        elif language == "en":
            return set(stopwords.words("english"))
        elif language == "fi":
            return set(stopwords.words("finnish"))
        elif language == "fr":
            return set(stopwords.words("french"))
        elif language == "de":
            return set(stopwords.words("german"))
        elif language == "hu":
            return set(stopwords.words("hungarian"))
        elif language == "it":
            return set(stopwords.words("italian"))
        elif language == "no":
            return set(stopwords.words("norwegian"))
        elif language == "pt":
            return set(stopwords.words("portuguese"))
        elif language == "ro":
            return set(stopwords.words("romanian"))
        elif language == "ru":
            return set(stopwords.words("russian"))
        elif language == "es":
            return set(stopwords.words("spanish"))
        elif language == "sv":
            return set(stopwords.words("swedish"))
        else:
            return []

# -*- coding: utf-8 -*-

