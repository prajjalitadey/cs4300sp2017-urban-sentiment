import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import matplotlib.pyplot as plt
from nltk.tokenize import TreebankWordTokenizer
import copy
from bs4 import UnicodeDammit

class Matrixizer:
    
    """
    variables
    self.matrix = tf-idf matrix
    self.vectorizer =  the vectorizer for everything
    self.dict_neighborhoods = gives the index where the neighborhood is at
    self.listing_review = 
    """
    
    def __init__(self, listing_reviews):
        
        self.vectorizer = TfidfVectorizer(max_df=.8, min_df=.05,
                                    max_features=5000, stop_words='english', norm='l2')
                
        #Making the tfidf matrix using our reviews data
        self.matrix = self.vectorizer.fit_transform([ listing.getReviews()  for _, listings in listing_reviews.iteritems() for listing in listings])

        #dictionary that maps listings to their id in doc_by_vocab matrix
        self.dict_neighborhoods = {neighborhood:[enumerate(i)] for neighborhood, listings in listings_reviews.iteritems() for listing in listings}
        
        
    
    def query(self, query):
        query_tfidf = tfidf_vec.transform([query])
        neighborhood_to_score = {}
        for neighborhood in dict_neighborhoods.keys():
            score = 0
            for listing in dict_neighborhood[neighborhood]:
                score += query_tfidf.multiply(matrix[listing])
            score = score / len(listing)
            neighborhood_to_score[neighborhood] =  score
        return neighborhood_to_score
        
    def bernoulli_bayes(self, test):
        vectorizer = CountVectorizer(ngram_range=(1, 2))  # for  unigrams only use ngram_range=(1, 1)
        vectorizer.fit(msgs_train)
        
        term_document_matrix_train = vectorizer.transform(msgs_train)
        term_document_matrix_train
        
        fsel=SelectKBest(score_func=chi2,k=1000)
        fsel.fit(term_document_matrix_train,classes_train)
        
        term_document_matrix_train=fsel.transform(term_document_matrix_train)
        
    
    def mutlinomial_bayes(self, test):
        pass
    
    def svm(self, test):
        pass
    
    