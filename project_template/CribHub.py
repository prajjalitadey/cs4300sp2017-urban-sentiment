from __future__ import print_function

import base64
import json
import urllib2
import math
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
from config import config
from numpy import linalg as la
import psycopg2



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
        file = urllib2.urlopen('https://s3.us-east-2.amazonaws.com/actual-cribhub/airbnb_tfidf_compressed.json')
        self.airbnb_tfidf_svd = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        #word_compressed to convert query idf to svd form (numpy array)
        file = urllib2.urlopen('https://s3.us-east-2.amazonaws.com/actual-cribhub/airbnb_words_compressed.json')
        self.airbnb_words_compressed = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        #word to index mapping to convert query to tfidf vector (dictionary)
        file = urllib2.urlopen('https://s3.us-east-2.amazonaws.com/actual-cribhub/airbnb_word_to_index.json')
        self.airbnb_word_to_index = json.load(file, encoding='utf8')
        # #neighborhood to listing_ids (dictionary)
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/neighborhood_to_listing_ids.json')
        self.neighborhood_to_listing_ids = json.load(file, encoding='utf8')

        for nbhd in self.neighborhood_to_listing_ids.keys():
            self.neighborhood_to_listing_ids[nbhd] = set(self.neighborhood_to_listing_ids[nbhd])


        #listing_id to neighborhood
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/listing_id_to_neighborhood.json')
        self.listing_id_to_neighborhood = json.load(file, encoding='utf8')
        #Airbnb listing_id to index in matrix
        file = urllib2.urlopen('https://s3.us-east-2.amazonaws.com/actual-cribhub/airbnb_id_to_idx.json')
        self.airbnb_id_to_idx = json.load(file, encoding='utf8')
        #index in matrix to the listing id it corresponds to
        # file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_idx_to_id.json')
        # self.airbnb_idx_to_id = json.load(file, encoding='utf8')
        #IDF values for all the words
        file = urllib2.urlopen('https://s3.us-east-2.amazonaws.com/actual-cribhub/airbnb_idf_values.json')
        self.airbnb_idf_values = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        #listing_id to listing dict
        #file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_listing_id_to_listing.json')
        #self.airbnb_lid_to_text = json.load(file, encoding='utf8')



        ########### NYTIMES DATA #############
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nytimes_tfidf_compressed.json')
        self.nytimes_tfidf_svd = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        #word_compressed to convert query idf to svd form (numpy array)
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nytimes_words_compressed.json')
        self.nytimes_words_compressed = json.load(file, object_hook=json_numpy_obj_hook, encoding='utf8')
        #word to index mapping to convert query to tfidf vector (dictionary)
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nytimes_word_to_index.json')
        self.nytimes_word_to_index = json.load(file, encoding='utf8')
        #neighborhood to review_ids (dictionary)
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nyt_nbhd_to_review.json')
        self.nytimes_nbhd_to_review = json.load(file, encoding='utf8')
        #listing_id to neighborhood
        file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/nyt_id_to_review.json')
        self.nytimes_id_to_review = json.load(file, encoding='utf8')
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

        #######DATABASES#######
        params = config()
        self.conn = psycopg2.connect(**params)
        self.cur = self.conn.cursor()
        # self.conn.autocommit(True)
        
        
        #######GLOBAL##########
        self.num_listings = 31121

        #######GLOBAL##########
        self.num_listings = 31121



    # function that vectorizes every review
    # iterate through them all, make the ids
    # list of listing ids

    #Gets out a list of tuples of form [(listing_id, review)]

    def generateAirbnbReviewVectors(self):
        review_dict = {}

        for nbhd in self.neighborhood_to_listing_ids.keys():
            nbhd_ids = self.neighborhood_to_listing_ids[nbhd]
            listing_ids = [x for x in nbhd_ids]

            for listing_id in listing_ids:
                text_output = self.get_text([listing_id])
                listing_text = text_output[0][1]
                reviews = listing_text.split("-----")
                for i in range(len(reviews)):
                    review_dict[str(listing_id)+"X"+str(i)] = np.array_str(self.get_query_svd(reviews[i], self.airbnb_word_to_index, self.airbnb_idf_values, self.airbnb_words_compressed))

        return review_dict

    def generateNYTReviewVectors(self):
        review_dict = {}

        for review_id in self.nytimes_id_to_review.keys():
            review_text = self.nytimes_id_to_review[review_id]
            review_dict[str(review_id)] = np.array_str(self.get_query_svd(review_text, self.nytimes_word_to_index, self.nytimes_idf_values, self.nytimes_words_compressed))

        return review_dict
            
    def get_listing_score(self, query_svd, listing_id):
        idx = self.airbnb_id_to_idx[listing_id]
        tfidf_svd = np.array(self.airbnb_tfidf_svd[idx])
        return tfidf_svd.dot(query_svd)


    def get_nyt_review_score(self, query_svd, review_id):
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


    def score_airbnb_neighborhoods(self, query, vec=False):
        if not vec:
            query_svd = self.get_query_svd(query, self.airbnb_word_to_index, self.airbnb_idf_values, self.airbnb_words_compressed)
        else:
            query_svd = query

        all_neighborhood_scores = defaultdict(list)
        for listing_id in self.airbnb_id_to_idx.keys():
            neighborhood = self.listing_id_to_neighborhood[listing_id]
            score_info = self.get_listing_score(query_svd, listing_id)
            all_neighborhood_scores[neighborhood].append((listing_id, score_info))

        if not vec:
            topic_neighborhoods = self.topic_modeling(query)
            ### If the query gives None in topic model it means it was not in the index.
            ### If this is the case, then it will just return the original scores
            ### Otherwise, the topic modeling will zero out the other scores.
            topic_neighborhood_scores = {}
            if topic_neighborhoods is None:
                topic_neighborhood_scores = all_neighborhood_scores
            else:
                for neighborhood in all_neighborhood_scores.keys():
                    if neighborhood in topic_neighborhoods:
                        topic_neighborhood_scores[neighborhood] = all_neighborhood_scores[neighborhood]
                    else:
                        topic_neighborhood_scores[neighborhood] = [(item[0], 1.0*item[1])for item in all_neighborhood_scores[neighborhood]]

        neighborhood_to_score = {}
        topic_neighborhood_scores = all_neighborhood_scores
        for neighborhood, scores in topic_neighborhood_scores.iteritems():
            score_avg = np.mean([score[1] for score in scores])
            neighborhood_to_score[neighborhood] = score_avg

        return neighborhood_to_score, topic_neighborhood_scores


    def score_nytimes_neighborhoods(self, query, vec=False):
        if not vec:
            query_svd = self.get_query_svd(query, self.nytimes_word_to_index, self.nytimes_idf_values, self.nytimes_words_compressed)
        else:
            query_svd = query

        all_neighborhood_scores = defaultdict(list)
        for review_id in self.nytimes_id_to_idx.keys():
            neighborhood = self.nytimes_id_to_neighborhood[review_id]
            score_info = self.get_nyt_review_score(query_svd, review_id)
            all_neighborhood_scores[neighborhood].append((review_id, score_info))

        neighborhood_to_score = {}
        for neighborhood, scores in all_neighborhood_scores.iteritems():  # scores is a list of tuples with id, score
            score_avg = np.mean([score[1] for score in scores])
            neighborhood_to_score[neighborhood] = score_avg

        return neighborhood_to_score


    def combine_scores(self, airbnb_scores, nytimes_scores):
        a = 0.8
        b = 0.2
        range1 = 0.01
        range2 = 0.05
        airbnb = set(airbnb_scores.keys())
        nytimes = set(nytimes_scores.keys())
        all_neighorhoods = airbnb.union(nytimes)
        both  = airbnb.intersection(nytimes)
        airbnb_only = airbnb - both
        nytimes_only = nytimes - both
    
        neighborhood_to_score = {}
    
        # neighborhood in both airbnb & nytimes -- CLEAN THIS
        for nbhd in both:
            count = len(set(self.neighborhood_to_listing_ids[nbhd]))

            neighborhood_to_score[nbhd] = a*airbnb_scores[nbhd]*math.log(count,6) + b*nytimes_scores[nbhd]

        # only airbnb neighborhoods
        for nbhd in airbnb_only:
            count = len(set(self.neighborhood_to_listing_ids[nbhd]))
            neighborhood_to_score[nbhd] = a*airbnb_scores[nbhd]*math.log(count,6)

        # only nytimes neighborhoods
        for nbhd in nytimes_only:
            neighborhood_to_score[nbhd] = b*nytimes_scores[nbhd]
            self.neighborhood_to_listing_ids[nbhd] = []

        # code to check whether there aren't any results
        empty_results = False
        if all(v == 0 for v in neighborhood_to_score.values()):
            empty_results = True
        return neighborhood_to_score, empty_results


    # new function
    def handle_query(self, query):

        query_criteria = query.split(",")
        if len(query_criteria) > 1:
            query_criteria.append(query)
        query_criteria = [q.strip() for q in query_criteria]
        # query_label = self.get_sentiment(query)['label']

        neighborhood_ranking = {}
        document_ranking = {}

        for criteria in query_criteria:
            # get neighborhood score
            a,listing_id_scores = self.score_airbnb_neighborhoods(criteria)
            b = self.score_nytimes_neighborhoods(criteria)
            scores, empty = self.combine_scores(a, b)

            nbhd_scores = sorted([[nbhd, score] for nbhd, score in scores.iteritems()], key=lambda x: x[1], reverse=True)
            nbhd_scores_enum = list(enumerate(nbhd_scores))
            nbhd_ranks = {nbhd[0]: rank for rank, nbhd in nbhd_scores_enum}
            listing_scores = []
            for neighborhood in listing_id_scores:
                listing_scores += listing_id_scores[neighborhood]

            listing_scores = sorted(listing_scores,key=lambda x: x[1], reverse=True)
            sorted_listingids = [x[0] for x in listing_scores]

            airbnb_query_svd = self.get_query_svd(criteria, self.airbnb_word_to_index, self.airbnb_idf_values, self.airbnb_words_compressed)
            listing_text = self.get_text(sorted_listingids[:3])
            airbnb_ranking = []
            if listing_text:
                for lid, text in listing_text:
                    # split into reviews

                    listing_score = self.get_listing_score(airbnb_query_svd, str(lid))
                    nbhd_rank = nbhd_ranks[self.listing_id_to_neighborhood[str(lid)]]
                    listofreviews = self.get_best_review_for_text(airbnb_query_svd, text)
                    for review, score in listofreviews:
                        airbnb_ranking.append(['airbnb', nbhd_rank, lid, score, review])

            # get all review scores
            query_svd = self.get_query_svd(criteria, self.nytimes_word_to_index, self.nytimes_idf_values, self.nytimes_words_compressed)
            review_ranking = []

            for rid, text in self.nytimes_id_to_review.iteritems():
                review_score = self.get_nyt_review_score(query_svd, rid)
                nbhd_rank = nbhd_ranks[self.nytimes_id_to_neighborhood[rid]]
                review_ranking.append(['nytimes', nbhd_rank, rid, 0.3*review_score, text])
            documents = sorted(airbnb_ranking + review_ranking, key=lambda x: x[3], reverse=True)[:5]

            # replace full listing text for best review, for airbnb docs
            # documents = [[doc[0], doc[1], doc[2], doc[3], re.sub('\\.', '', self.get_best_review_for_text(airbnb_query_svd, doc[4])[0])] if doc[0] is 'airbnb' else [doc[0], doc[1], doc[2], doc[3], re.sub('\\.', '', doc[4])] for doc in documents]
            if len(query_criteria) == 1:
                neighborhood_ranking["all_criteria"] = nbhd_scores
                document_ranking["all_criteria"] = documents

            elif criteria is query:
                criteria = 'all_criteria'
            neighborhood_ranking[criteria] = nbhd_scores
            document_ranking[criteria] = documents

        return {'neighborhood_ranking': neighborhood_ranking, 'document_ranking': document_ranking, 'query': query, 'empty': empty}




    # rel & irr is in form ['airbnb', id#]
    def rocchio(self, q, rel, irr, a=0, b=15, c=10, clip=False):
        airbnb_wt = 0.8
        nytimes_wt = 0.2

        # separate airbnb & nytimes relevant and irrelevant docs
        airbnb_rel = [str(lid) for lid in rel if str(lid) in self.airbnb_id_to_idx.keys()]
        airbnb_irr = [str(lid) for lid in irr if str(lid) in self.airbnb_id_to_idx.keys()]
        nytimes_rel = [str(rid) for rid in rel if str(rid) in self.nytimes_id_to_idx.keys()]
        nytimes_irr = [str(rid) for rid in irr if str(rid) in self.nytimes_id_to_idx.keys()]

        airbnb_query_vec = self.get_query_svd(q, self.airbnb_word_to_index, self.airbnb_idf_values, self.airbnb_words_compressed)
        airbnb_q_mod = a*airbnb_query_vec

        # airbnb list of vectors
        airbnb_rel_vecs = [self.airbnb_tfidf_svd[self.airbnb_id_to_idx[aid]] for aid in airbnb_rel]
        if len(airbnb_rel_vecs) > 0:
            airbnb_rel_avg = np.array(airbnb_rel_vecs).mean(0)
            airbnb_q_mod += b*airbnb_rel_avg
        airbnb_irr_vecs = [self.airbnb_tfidf_svd[self.airbnb_id_to_idx[aid]] for aid in airbnb_irr]
        if len(airbnb_irr_vecs) > 0:
            airbnb_irr_avg = np.array(airbnb_irr_vecs).mean(0)
            airbnb_q_mod -= c*airbnb_irr_avg


        nytimes_query_vec = self.get_query_svd(q, self.nytimes_word_to_index, self.nytimes_idf_values, self.nytimes_words_compressed)
        nytimes_q_mod = a*nytimes_query_vec

        # nytimes list of vectors
        nytimes_rel_vecs = [self.nytimes_tfidf_svd[self.nytimes_id_to_idx[nid]] for nid in nytimes_rel]
        if len(nytimes_rel_vecs) > 0:
            nytimes_rel_avg = np.array(nytimes_rel_vecs).mean(0)
            nytimes_q_mod += b*nytimes_rel_avg
        nytimes_irr_vecs = [self.nytimes_tfidf_svd[self.nytimes_id_to_idx[nid]] for nid in nytimes_irr]
        if len(nytimes_irr_vecs) > 0:
            nytimes_irr_avg = np.array(nytimes_irr_vecs).mean(0)
            nytimes_q_mod -= c*nytimes_irr_avg


        if clip:
            for abnb_weight in np.nditer(airbnb_q_mod, op_flags=['readwrite']):
                if abnb_weight < 0:
                    abnb_weight[...] = 0
            for nyt_weight in np.nditer(airbnb_q_mod, op_flags=['readwrite']):
                if nyt_weight < 0:
                    nyt_weight[...] = 0

        ###### handling query again
        neighborhood_ranking = {}
        document_ranking = {}

        a, listing_id_scores = self.score_airbnb_neighborhoods(airbnb_q_mod, True)
        b = self.score_nytimes_neighborhoods(nytimes_q_mod, True)
        scores, empty = self.combine_scores(a, b)


        nbhd_scores = sorted([[nbhd, score] for nbhd, score in scores.iteritems()], key=lambda x: x[1], reverse=True)
        nbhd_scores_enum = list(enumerate(nbhd_scores))
        nbhd_ranks = {nbhd[0]: rank for rank, nbhd in nbhd_scores_enum}
        listing_scores = []
        for neighborhood in listing_id_scores:
            listing_scores += listing_id_scores[neighborhood]

        listing_scores = sorted(listing_scores,key=lambda x: x[1], reverse=True)
        sorted_listingids = [x[0] for x in listing_scores]

        # listing_ids = self.airbnb_id_to_idx.keys()
        listing_text = self.get_text(sorted_listingids[:5])

        airbnb_ranking = []
        if listing_text:
            for lid, text in listing_text:
                listing_score = self.get_listing_score(airbnb_q_mod, str(lid))
                nbhd_rank = nbhd_ranks[self.listing_id_to_neighborhood[str(lid)]]
                airbnb_ranking.append(['airbnb', nbhd_rank, lid, listing_score, text])

        # get all review scores
        review_ranking = []
        for rid, text in self.nytimes_id_to_review.iteritems():
            review_score = self.get_nyt_review_score(nytimes_q_mod, rid)
            nbhd_rank = nbhd_ranks[self.nytimes_id_to_neighborhood[rid]]
            review_ranking.append(['nytimes', nbhd_rank, rid, 0.3*review_score, text])

        #documents = sorted(listing_ranking + review_ranking, key=lambda x: x[1])
        documents = sorted(airbnb_ranking + review_ranking, key=lambda x: x[3], reverse=True)[:5]
        documents = [[doc[0], doc[1], doc[2], doc[3], re.sub('\\.', '', self.get_best_review_for_text(airbnb_query_svd, doc[4])[0])] if doc[0] is 'airbnb' else [doc[0], doc[1], doc[2], doc[3], re.sub('\\.', '', doc[4])] for doc in documents]

        # if criteria is query:
        #     criteria = 'all_criteria'
        neighborhood_ranking[q] = nbhd_scores
        document_ranking[q] = documents

        return {'neighborhood_ranking': neighborhood_ranking, 'document_ranking': document_ranking, 'query': q, 'empty': empty}



    # Returns nothing if word is not found in topic model
    def topic_modeling(self, query):
        query_words = query.split(" ")
        indexes = [self.word_to_top_index[q] for q in query_words if q in self.word_to_top_index]
        if not indexes:
            return None

        #Adding all the query words together
        vec = np.zeros(10)
        for topic in indexes:
            vec += self.topic_matrix[:, topic]
        #Checking if all values in vector are same if so we retun None
        topic = str(np.argmax(vec))
        return self.topic_to_neighborhoods[topic]


    # Returns sentiment from Lillian Lee's text-processing API: http://text-processing.com/docs/sentiment.html
    # Input: text string
    # Output: dictionary
    # Output structure: {'probability':{u'neg':0.0222, u'neutral':0.1134, u'pos':0.7183}, u'label':u'pos'}
    def get_sentiment(self, content):
        r = requests.post("http://text-processing.com/api/sentiment/", {"text": content})
        results = json.loads(r.text)
        return results

    # Put in a list of listing_ids you want to get text for
    # Gets out a list of tuples of form [(listing_id, review)]
    # If seperated is True then it returns
    def get_text(self, listing_ids, separated=False):
        """ query parts from the parts table """
        try:
            placeholders = ", ".join(str(lid) for lid in listing_ids)
            query = "SELECT * FROM listingid_to_text_sep WHERE listing_id IN (%s)" % placeholders
            self.cur.execute(query)
            rows = self.cur.fetchall()
            if separated:
                rows = [(row[0], row[1].split('-----')) for row in rows]
            return rows
        except (Exception, psycopg2.DatabaseError) as error:
            self.conn.rollback()

    def get_neighborhood_information(self, query, neighborhood):
            print(query)
            query_svd = self.get_query_svd(query, self.airbnb_word_to_index, self.airbnb_idf_values, self.airbnb_words_compressed)
            print("-" * 30)
            print(query_svd)
            print("-" * 30)
            listing_ids = self.neighborhood_to_listing_ids[neighborhood]
            listings_to_score = [(listing, self.get_listing_score(query_svd, str(listing))) for listing in listing_ids]
            best_five, _ = zip(*sorted(listings_to_score, key = lambda x: x[1], reverse = True)[:5])
            all_reviews = self.get_text(best_five, separated=True)
            #sentiment_reviews = [(listing, self.get_sentiment(review), review) for listing, reviews in all_reviews for review in reviews]
            #sorted_sentiment_reviews = sorted(sentiment_reviews, key = lambda x: x[1], reverse = True)
            reviews_svd = [(listing, self.get_query_svd(review, self.airbnb_word_to_index, self.airbnb_idf_values, self.airbnb_words_compressed), review)
                           for listing, reviews in all_reviews for review in reviews]
            review_scores = [(listingid, query_svd.dot(review_svd) / la.norm(review_svd), review) for listingid, review_svd, review in reviews_svd]
            top_airbnb_reviews = sorted(review_scores, key = lambda x: x[1], reverse = True)[:10]

            query_svd = self.get_query_svd(query, self.nytimes_word_to_index, self.nytimes_idf_values, self.nytimes_words_compressed)
            ny_review_ids =  self.nytimes_nbhd_to_review[neighborhood]
            ny_review_svd_str = self.get_nyt_review_scores(ny_review_ids)
            ny_review_svd = [(listing_id, np.fromstring(review_svd, dtype='float64', count=50)) for listing_id, review_svd in ny_review_svd_str]
            ny_review_scores = [(listingid, query_svd.dot(review_svd) / la.norm(review_svd), self.nytimes_id_to_review[str(listingid)]) for listingid, review_svd in ny_review_svd]
            top_nyt_reviews = sorted(ny_review_scores, key = lambda x: x[1], reverse = True)[:10]

            top_reviews = top_airbnb_reviews
            if (top_reviews[0][1] == 0.0):
                top_reviews = top_nyt_reviews
            print(top_nyt_reviews)

            return top_reviews #, sorted_sentiment_reviews[10:], sorted_sentiment_reviews[:10]


    def get_best_review_for_text(self, query_svd, text):
        reviews = text.split("-----")
        ####################### REPLACE FOLLOWING LINE, CALL DATABASE INSTEAD #######################
        reviews_svd = [(review, self.get_query_svd(review, self.airbnb_word_to_index, self.airbnb_idf_values, self.airbnb_words_compressed)) for review in reviews]
        review_scores = [(review, query_svd.dot(review_svd)) for review, review_svd in reviews_svd]
        top_review = sorted(review_scores, key=lambda x: x[1], reverse=True)[:4]
        return top_review

    def get_nyt_review_scores(self, nyt_reviews):
        conn = None
        placeholders = ", ".join(str(rid) for rid in nyt_reviews)
        print(placeholders)
        query = "SELECT * FROM nytimes_review_to_vec WHERE review_id IN (%s)" % placeholders
        try:
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            print("-" * 10)
            print(rows)
            return rows
        except (Exception, psycopg2.DatabaseError) as error:
            self.conn.rollback()
