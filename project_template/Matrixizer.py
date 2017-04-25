import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


class Matrixizer:

    """
    variables
    self.matrix = tf-idf matrix
    self.vectorizer =  the vectorizer for everything
    self.dict_neighborhoods = gives the index where the neighborhood is at
    self.listing_review =
    """

    def __init__(self, listing_reviews):
        self.vectorizer = TfidfVectorizer(max_df=1.0, min_df=0.0,
                                          max_features=5000, stop_words='english', norm='l2')

        #Making the tfidf matrix using our reviews data

        try: 
            self.matrix = self.vectorizer.fit_transform([listing.getReviews() for _, listings in listing_reviews.iteritems() for listing in listings])
        except:
            print("AHAJKJFDLK:JFLKADJFLK:DJGLKDJAFLK:JDA_-----------------------------------------")
            print([listing.getReviews() for _, listings in listing_reviews.iteritems() for listing in listings])


        #dictionary that maps listings to their id in doc_by_vocab matrix
        i = 0
        self.dict_neighborhoods = defaultdict(list)
        for neighborhood, listings in listing_reviews.iteritems():
            for listing in listings:
                self.dict_neighborhoods[neighborhood].append(i)
                i += 1

    def query(self, query):
        query_tfidf = self.vectorizer.transform([query]).toarray().T
        neighborhood_to_score = {}
        for neighborhood in self.dict_neighborhoods.keys():
            score = 0
            for listing in self.dict_neighborhoods[neighborhood]:
                score += self.matrix.getrow(listing).dot(query_tfidf)[0]
            score = float(score) / len(self.dict_neighborhoods[neighborhood])
            neighborhood_to_score[neighborhood] = score
            print(score)
        return neighborhood_to_score

    def bernoulli_bayes(self, test):
        vectorizer = TfidfVectorizer.CountVectorizer(ngram_range=(1, 2))  # for  unigrams only use ngram_range=(1, 1)
        vectorizer.fit(msgs_train)

        term_document_matrix_train = vectorizer.transform(msgs_train)
        term_document_matrix_train

        fsel = TfidfVectorizer.SelectKBest(score_func=chi2, k=1000)
        fsel.fit(term_document_matrix_train,classes_train)

        term_document_matrix_train = fsel.transform(term_document_matrix_train)

    def mutlinomial_bayes(self, test):
        pass

    def svm(self, test):
        pass
