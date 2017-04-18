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
        
        self.vectorizer = TfidfVectorizer(max_df=Max_df, min_df=Min_df,
                                    max_features=n_feats,stop_words='english', norm='l2')
                
        #Making the tfidf matrix using our reviews data
        self.matrix = self.vectorizer.fit_transform(listings_reviews.values())

        #dictionary that maps neighborhoods to their id in doc_by_vocab matrix
        self.dict_neighborhoods = {neighborhood:enumerate(i) for neighborhood in listings_reviews}
        
        
    
    def query(self, query):
        query_tfidf = tfidf_vec.transform([query])
        return {neighborhood:query_tfidf.multiply(doc_by_vocab[dict_neighborhoods[neighborhood]])}
        
    def bernoulli_bayes(self, ):
        vectorizer = CountVectorizer(ngram_range=(1, 2))  # for  unigrams only use ngram_range=(1, 1)
        vectorizer.fit(msgs_train)
        
        term_document_matrix_train = vectorizer.transform(msgs_train)
        term_document_matrix_train
        
        fsel=SelectKBest(score_func=chi2,k=1000)
        fsel.fit(term_document_matrix_train,classes_train)
        
        term_document_matrix_train=fsel.transform(term_document_matrix_train)
        
    
    def mutlinomial_bayes(self, ):
        pass
    
    def svm(self, ):
        pass
    
    def (self. ):
        pass
    
    
    