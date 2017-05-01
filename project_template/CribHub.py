from __future__ import print_function

import base64
import json
import urllib2
# from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict
# from collections import Counter
# from numpy import linalg as LA
# from scipy.sparse.linalg import svds
# from sklearn.preprocessing import normalize
# import operator
# import io
# import math
import re


class CribHub:
    """
    self.airbnb_tfidf         ---    tf-idf matrix for airbnb data
    self.nytimes_tfidf        ---    tf-idf matrix
    self.airbnb_vectorizer    ---    vectorizer for airbnb data
    self.nytimes_vectorizer   ---    vectorizer for nytimes data
    self.airbnb_dict          ---    dictionary of neighborhood to listings
    self.nytimes_dict         ---    dictionary of neighborhood to reviews
    """

    def __init__(self):
        class NumpyEncoder(json.JSONEncoder):

            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    if obj.flags['C_CONTIGUOUS']:
                        obj_data = obj.data
                    else:
                        cont_obj = np.ascontiguousarray(obj)
                        assert(cont_obj.flags['C_CONTIGUOUS'])
                        obj_data = cont_obj.data
                    data_b64 = base64.b64encode(obj_data)
                    return dict(__ndarray__=data_b64,
                                dtype=str(obj.dtype),
                                shape=obj.shape)
                # Let the base class default method raise the TypeError
                return json.JSONEncoder(self, obj)

        def json_numpy_obj_hook(dct):
            if isinstance(dct, dict) and '__ndarray__' in dct:
                data = base64.b64decode(dct['__ndarray__'])
                return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
            return dct


        ########### AIRBNB DATA #############
        #airbnb tfidf matrix after svd (numpy array)
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_tfidf_compressed.json')
        self.airbnb_tfidf_svd = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        #word_compressed to convert query idf to svd form (numpy array)
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_words_compressed.json')
        self.airbnb_words_compressed = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        #word to index mapping to convert query to tfidf vector (dictionary)
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_word_to_index.json')
        self.airbnb_word_to_index = json.load(file, encoding='utf8')
        # #neighborhood to listing_ids (dictionary)
        # file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/neighborhood_to_listing_ids.json')
        # self.neighborhood_to_listing_ids = json.load(file, encoding='utf8')
        #listing_id to neighborhood
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/listing_id_to_neighborhood.json')
        self.listing_id_to_neighborhood = json.load(file, encoding='utf8')
        #Airbnb listing_id to index in matrix
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_id_to_idx.json')
        self.airbnb_id_to_idx = json.load(file, encoding='utf8')
        #index in matrix to the listing id it corresponds to
        # file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_idx_to_id.json')
        # self.airbnb_idx_to_id = json.load(file, encoding='utf8')
        #IDF values for all the words
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_idf_values.json')
        self.airbnb_idf_values = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        #listing_id to listing dict
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_listing_id_to_listing.json')
        self.airbnb_lid_to_text = json.load(file, encoding='utf8')



        ########### NYTIMES DATA #############
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nytimes_tfidf_compressed.json')
        self.nytimes_tfidf_svd = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        #word_compressed to convert query idf to svd form (numpy array)
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nytimes_words_compressed.json')
        self.nytimes_words_compressed = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        #word to index mapping to convert query to tfidf vector (dictionary)
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nytimes_word_to_index.json')
        self.nytimes_word_to_index = json.load(file, encoding='utf8')
        #neighborhood to listing_ids (dictionary)
        # file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nyt_nbhd_to_review.json')
        # self.nytimes_nbhd_to_review = json.load(file, encoding='utf8')
        #listing_id to neighborhood
        # file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nyt_id_to_review.json')
        # self.nytimes_id_to_review = json.load(file, encoding='utf8')
        #Airbnb listing_id to index in matrix
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nyt_id_to_idx.json')
        self.nytimes_id_to_idx = json.load(file, encoding='utf8')
        #index in matrix to the listing id it corresponds to
        # file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nyt_idx_to_id.json')
        # self.nytimes_idx_to_id = json.load(file, encoding='utf8')
        #IDF values for all the words
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nytimes_idf_values.json')
        self.nytimes_idf_values = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        # NYT id to neighborhood dictionary
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nyt_id_to_neighborhood.json')
        self.nytimes_id_to_neighborhood = json.load(file, encoding='utf8')





    def get_listing_score(self, query_svd, listing_id):
        idx = self.airbnb_id_to_idx[listing_id]
        tfidf_svd = np.array(self.airbnb_tfidf_svd[idx])
        return tfidf_svd.dot(query_svd)


    def get_review_score(self, query_svd, review_id):
        idx = self.nytimes_id_to_idx[review_id]
        tfidf_svd = np.array(self.nytimes_tfidf_svd[idx])
        return tfidf_svd.dot(query_svd)


    def get_query_svd(self, query, word_to_index, idf_values, words_compressed):
        query_tfidf = np.zeros(len(word_to_index))

        for word in re.sub("[^\w]", " ", query).split():
            # should take into account words that don't exist in dictionary, maybe by smoothing
            word = word.replace(",", "")
            if word in word_to_index.keys():
                word_idx = word_to_index[word]  # word index
                query_tfidf[word_idx] = 1.0 * idf_values[word_idx]  # set that word to 1 times its idf value
        words_transpose = words_compressed.T
        query_tfidf_T = np.matrix(query_tfidf).T
        query_svd = words_transpose * query_tfidf_T  # convert to svd form
        denominator = float(np.linalg.norm(query_svd))
        if denominator == 0:
            query_svd = query_svd/1
        else:
            query_svd = query_svd/denominator

        return np.array(query_svd.T)[0]


    def score_airbnb_neighborhoods(self, query):
        query_svd = self.get_query_svd(query, self.airbnb_word_to_index, self.airbnb_idf_values, self.airbnb_words_compressed)
        all_neighborhood_scores = defaultdict(list)  # dict= neighborhood:[(listing1,score1), (listing2,score2)...]

        for listing_id in self.airbnb_id_to_idx.keys():  # going through all the listing ids
            neighborhood = self.listing_id_to_neighborhood[listing_id]
            score_info = self.get_listing_score(query_svd, listing_id)
            all_neighborhood_scores[neighborhood].append((listing_id, score_info))

        neighborhood_to_score = {}
        for neighborhood, scores in all_neighborhood_scores.iteritems():  # scores is a list of tuples with id, score
            score_avg = np.mean([score[1] for score in scores])
            neighborhood_to_score[neighborhood] = score_avg

        return neighborhood_to_score


    def score_nytimes_neighborhoods(self, query):
        query_svd = self.get_query_svd(query, self.nytimes_word_to_index, self.nytimes_idf_values, self.nytimes_words_compressed)
        all_neighborhood_scores = defaultdict(list)

        for review_id in self.nytimes_id_to_idx.keys():
            neighborhood = self.nytimes_id_to_neighborhood[review_id]
            score_info = self.get_review_score(query_svd, review_id)
            all_neighborhood_scores[neighborhood].append((review_id, score_info))

        neighborhood_to_score = {}
        for neighborhood, scores in all_neighborhood_scores.iteritems():  # scores is a list of tuples with id, score
            score_avg = np.mean([score[1] for score in scores])
            neighborhood_to_score[neighborhood] = score_avg

        return neighborhood_to_score



    def combine_scores(self, airbnb_scores, nytimes_scores):
        a = 0.8
        b = 0.2

        airbnb = set(airbnb_scores.keys())
        nytimes = set(nytimes_scores.keys())

        both = airbnb.intersection(nytimes)
        airbnb_only = airbnb - both
        nytimes_only = nytimes - both

        neighborhood_to_score = {}

        # neighborhood in both airbnb & nytimes
        for nbhd in both:
            neighborhood_to_score[nbhd] = a*airbnb_scores[nbhd] + b*nytimes_scores[nbhd]
        # only airbnb neighborhoods
        for nbhd in airbnb_only:
            neighborhood_to_score[nbhd] = a*airbnb_scores[nbhd]
        # only nytimes neighborhoods
        for nbhd in nytimes_only:
            neighborhood_to_score[nbhd] = b*nytimes_scores[nbhd]

        return neighborhood_to_score




    def handle_query(self, query):
        query_criteria = query.split(",")
        query_criteria.append(query)
        query_criteria = [q.strip() for q in query_criteria]


        neighborhood_ranking = {}
        listing_ranking = defaultdict(list)

        for criteria in query_criteria:
            # get neighborhood score
            scores = self.combine_scores(self.score_airbnb_neighborhoods(criteria), self.score_nytimes_neighborhoods(criteria))
            nbhd_scores_list = sorted([[nbhd, score] for nbhd, score in scores.iteritems()], key=lambda x: x[1], reverse=True)
            neighborhood_ranking[criteria] = nbhd_scores_list

            # get all listing scores
            query_svd = self.get_query_svd(criteria, self.airbnb_word_to_index, self.airbnb_idf_values, self.airbnb_words_compressed)
            for lid, text in self.airbnb_lid_to_text.iteritems():
                listing_score = self.get_listing_score(query_svd, lid)
                listing_ranking[criteria].append([lid, listing_score, text])

            listing_ranking[criteria] = sorted(listing_ranking[criteria], key=lambda x: x[1], reverse=True)

        return {'neighborhood_ranking': neighborhood_ranking, 'listing_ranking': listing_ranking, 'query': query}



    def rocchio(self, q, airbnb_rel, airbnb_irr, nytimes_rel, nytimes_irr, a=.7, b=.5, c=.5, clip=True):
        airbnb_wt = 0.8
        nytimes_wt = 0.2

        airbnb_rel_vecs = [self.airbnb_tfidf_svd[self.airbnb_id_to_idx[aid]] for aid in airbnb_rel]
        airbnb_rel_avg = np.array(airbnb_rel_vecs).mean(0)
        airbnb_irr_vecs = [self.airbnb_tfidf_svd[self.airbnb_id_to_idx[aid]] for aid in airbnb_irr]
        airbnb_irr_avg = np.array(airbnb_irr_vecs).mean(0)
        nytimes_rel_vecs = [self.nytimes_tfidf_svd[self.nytimes_id_to_idx[nid]] for nid in nytimes_rel]
        nytimes_rel_avg = np.array(nytimes_rel_vecs).mean(0)
        nytimes_irr_vecs = [self.nytimes_tfidf_svd[self.nytimes_id_to_idx[nid]] for nid in nytimes_irr]
        nytimes_irr_avg = np.array(nytimes_irr_vecs).mean(0)

        rel_avg = airbnb_wt*airbnb_rel_avg + nytimes_wt*nytimes_rel_avg
        irrel_avg = airbnb_wt*airbnb_irr_avg + nytimes_wt*nytimes_irr_avg

        query_vec = self.get_query_svd(q, self.airbnb_word_to_index, self.airbnb_idf_values, self.airbnb_words_compressed)
        q_mod = a*query_vec + b*rel_avg - c*irrel_avg

        if clip:
            for weight in np.nditer(q_mod, op_flags=['readwrite']):
                if weight < 0:
                    weight[...] = 0

        return self.handle_query(q_mod)


# if __name__ == '__main__':
#     query = "port authority"

#     cribhub = CribHub()
#     print ("AWS Loaded")
#     m = cribhub.handle_query(query)
