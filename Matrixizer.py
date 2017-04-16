import sys
#import textmining
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import matplotlib.pyplot as plt
from nltk.tokenize import TreebankWordTokenizer
import copy
from bs4 import UnicodeDammit

class Matrixizer:
    
    """
    #code
    self.matrix = tf-idf matrix
    self.vectorizer =  the vectorizer for everything
    self.dict_neighborhoods = gives the index where the neighborhood is at
    """
    
    def __init__(self, n_feats, Max_df, Min_df):
        
        #Parse the values from the built Parser class
        parser = Parser()
        reviews, listings = parser.getcsvs()
        listings_reviews = parser.getcsvs(reviews, listings)
            
        self.vectorizer = TfidfVectorizer(max_df=Max_df, min_df=Min_df,
                                    max_features=n_feats,stop_words='english', norm='l2')
        
        #Making the tfidf matrix using our reviews data
        self.matrix = self.vectorizer.fit_transform(scripts)

        #dictionary that maps neighborhoods to their id in doc_by_vocab matrix
        self.dict_neighborhoods = {neighborhood:enumerate(i) for neighborhood in listings_reviews}
        
    
    def query(self, query):
        query_tfidf = tfidf_vec.transform([query])
        return {neighborhood:query_tfidf.multiply(doc_by_vocab[dict_neighborhoods[neighborhood]])}
        
    def save(self):
        pass
    
    def query(self, ):
        pass
    
    def name(self, ):
        pass
    
    