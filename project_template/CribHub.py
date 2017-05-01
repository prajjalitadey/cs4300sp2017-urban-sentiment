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
        self.airbnb_lid_to_text = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')



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
        self.nytimes_id_to_neighborhood = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')

        ###TOPIC MODELING###
    
        # Matrix for Topic Modeling
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/topic_matrix.json')
        self.topic_matrix = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        # Maps the word to the col number that  corresponds to it in the Topic matrix above
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/word_to_top_index.json')
        self.word_to_top_index = json.load(file, encoding='utf8')
        # Maps the topic to the neighborhoods associated with that topic
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/topic_to_neighborhood.json')
        self.topic_to_neighborhoods = json.load(file, encoding='utf8')


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



    def rocchio(self, q, airbnb_relevant_ids, airbnb_irrelevant_ids, nytimes_relevant_ids, nytimes_irrelevant_ids, a=.7, b=.5, c=.5, clip=True):
        avg_rel_airbnb = np.zeros(len(airbnb_idf_values))
        avg_irrel_airbnb = np.zeros(len(airbnb_idf_values))

        for airbnb_id in airbnb_relevant_ids:
            rel_airbnb_idx = self.airbnb_id_to_idx[airbnb_id]
            avg_rel_airbnb += self.airbnb_tfidf_svd[rel_airbnb_idx]
        avg_rel_airbnb/= len(airbnb_relevant_ids)

        for airbnb_id in airbnb_irrelevant_ids:
            irrel_airbnb_idx = self.airbnb_id_to_idx[airbnb_id]
            avg_irrel_airbnb += self.airbnb_tfidf_svd[irrel_airbnb_idx]
        avg_irrel_airbnb /= len(airbnb_irrelevant_ids)


        avg_rel_nytimes = np.zeros(len(nytimes_idf_values))
        avg_irrel_nytimes = np.zeros(len(nytimes_idf_values))

        for nytimes_id in nytimes_relevant_ids:
            rel_nytimes_idx = self.nytimes_id_to_idx[nytimes_id]
            avg_rel_nytimes += self.nytimes_tfidf_svd[rel_nytimes_idx]
        avg_rel_nytimes /= len(nytimes_relevant_ids)

        for nytimes_id in nytimes_irrelevant_ids:
            irrel_nytimes_idx = self.nytimes_id_to_idx[nytimes_id]
            avg_irrel_nytimes += self.nytimes_tfidf_svd[irrel_nytimes_idx]
        avg_irrel_nytimes /= len(nytimes_irrelevant_ids)

        airbnb_wt = 0.8
        nytimes_wt = 0.2

        rel_avg = airbnb_wt*avg_rel_airbnb + nytimes_wt*avg_rel_nytimes
        irrel_avg = airbnb_wt*avg_irrel_airbnb + nytimes_wt*avg_irrel_nytimes

        q_new = a*q + b*rel_avg - c*irrel_avg

        return self.handle_query(q_new)


    # Returns nothing if word is not found in topic model
    def topic_modeling(self, query):
        query_words = query.split(" ")
        indexes = [word_to_top_index[q] for q in query_words if q in word_to_top_index]
        
        #Adding all the query words together
        vec = np.zeroes(10)
        for topic in indexes:
            vec += topic_matrix[:, indexes]
       
        #Checking if all values in vector are same if so we retun None
        usetopic = False
        value = vec[0]
        for i in range(len(vec)):
            if vec[0] != value:
                usetopic = True
        if usetopic:
            topic = np.argmax(vec)
            return topic_to_neighborhoods[topic]
        else:
            return None


        #     rel_airbnb_idx= airbnb_id_to_idx[relevant_id]
        #     relevant_svd=airbnb_tfidf_svd[rel_airbnb_idx]
        #     irrel_airbnb_idx=airbnb_id_to_idx[irrelevant_id]
        #     irrelevant_svd=airbnb_tfidf_svd[irrel_airbnb_idx]


        # for relevant_doc in relevant:
        #     relevant_tfidf = self.airbnb_vectorizer.transform(relevant_doc)
        #     relevant_avg_tfidf += relevant_tfidf
        # relevant_avg_tfidf = relevant_avg_tfidf/len(relevant)

        # for irrelevant_doc in irrelevant:
        #     irrelevant_tfidf = self.airbnb_vectorizer.transform(irrelevant_doc)
        #     irrelevant_avg_tfidf += irrelevant_tfidf
        # irrelevant_avg_tfidf = irrelevant_avg_tfidf/len(irrelevant)

        # q_new = c*q_tfidf+a*relevant_avg_tfidf-b*irrelevant_avg_tfidf
        # return q_new


if __name__ == '__main__':
    query = "port authority"

    cribhub = CribHub()
    m = cribhub.handle_query(query)
