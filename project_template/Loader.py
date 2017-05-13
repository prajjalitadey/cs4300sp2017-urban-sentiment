import urllib2
import json
import pandas as pd
from collections import defaultdict
from bs4 import UnicodeDammit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import numpy as np
import base64
import nltk

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict
            holding dtype, shape and the data, base64 encoded.
            """
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
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.
                :param dct: (dict) json encoded ndarray
                :   return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

##### LOAD DATA FROM S3 #####
def load_data():
    file1 = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_dict.json')
    airbnb_dictionary = json.load(file1, encoding='utf8')
    return {"airbnb_dictionary": airbnb_dictionary}


# airbnb function
def neighborhood_to_listing_ids():
    data_frame = pd.DataFrame.from_csv("/Users/prajjalitadey/Documents/Spring2017/4300/cs4300sp2017-urban-sentiment/jsons/nyc_combination.csv", index_col=None)
    # data_frame = pd.DataFrame.from_csv("/Users/prajjalitadey/Documents/Spring2017/4300/cs4300sp2017-urban-sentiment/jsons/test.csv", index_col=None)

    i = 0
    airbnb_dict = defaultdict(dict)
    neighborhood_to_listing_ids = defaultdict(list)

    for row in data_frame.itertuples():
        review = getattr(row, 'comments')
        if isinstance(review, basestring):
            review = UnicodeDammit(review).unicode_markup.encode("utf-8")
            neighborhood = (getattr(row, 'neighbourhood')).lower()
            listing_id = getattr(row, 'listing_id')
            neighborhood_to_listing_ids[neighborhood].append(listing_id)

        if(i % 1000 == 0):
            print(str(i)+" completed.")
        i += 1

    new_dict = {neighborhood: list(set(listing_ids)) for neighborhood, listing_ids in neighborhood_to_listing_ids.iteritems()}
    return new_dict


# airbnb function
def listing_id_to_listing():
    print("Listing_ID_to_listing Function")
    data_frame = pd.DataFrame.from_csv("/Users/prajjalitadey/Documents/Spring2017/4300/cs4300sp2017-urban-sentiment/jsons/nyc_combination.csv", index_col=None)
    # data_frame = pd.DataFrame.from_csv("/Users/prajjalitadey/Documents/Spring2017/4300/cs4300sp2017-urban-sentiment/jsons/test.csv", index_col=None)

    i = 0
    listing_id_to_listing_temp = defaultdict(list)
    for row in data_frame.itertuples():
        review = getattr(row, 'comments')
        if isinstance(review, basestring):
            review = " ".join(w for w in nltk.wordpunct_tokenize(review) if w.lower() in words or not w.isalpha())
            re.sub('\\.', '', review)
            review = UnicodeDammit(review).unicode_markup.encode("utf-8")
            neighborhood = (getattr(row, 'neighbourhood')).lower()
            listing_id = getattr(row, 'listing_id')

            listing_id_to_listing_temp[listing_id].append(review)

        if(i % 1000 == 0):
            print(str(i)+" completed.")
        i += 1

    # listing_id: a string that is the combination of all reviews for that listing
    listing_id_to_listing = {lid: '--ENDREVIEW--'.join(str_list) for lid, str_list in listing_id_to_listing_temp.iteritems()}

    return listing_id_to_listing


# airbnb function
def listing_id_to_listing_db():
    print("Listing_ID_to_listing Function")
    data_frame = pd.DataFrame.from_csv("/Users/prajjalitadey/Documents/Spring2017/4300/cs4300sp2017-urban-sentiment/jsons/nyc_combination.csv", index_col=None)
    # data_frame = pd.DataFrame.from_csv("/Users/prajjalitadey/Documents/Spring2017/4300/cs4300sp2017-urban-sentiment/jsons/test.csv", index_col=None)

    i = 0
    listing_id_to_listing_temp = defaultdict(list)
    for row in data_frame.itertuples():
        review = getattr(row, 'comments')
        if isinstance(review, basestring):
            re.sub('\\.', '', review)
            review = UnicodeDammit(review).unicode_markup.encode("utf-8")
            neighborhood = (getattr(row, 'neighbourhood')).lower()
            listing_id = getattr(row, 'listing_id')

            listing_id_to_listing_temp[listing_id].append(review)

        if(i % 1000 == 0):
            print(str(i)+" completed.")
        i += 1

    # listing_id: a string that is the combination of all reviews for that listing
    listing_id_to_listing = {lid: '--ENDREVIEW--'.join(str_list) for lid, str_list in listing_id_to_listing_temp.iteritems()}

    return listing_id_to_listing


# nyt function
def nyt_id_to_review():
    print("NYT_ID_to_Review Function")
    data_frame = pd.DataFrame.from_csv("/Users/prajjalitadey/Documents/Spring2017/4300/cs4300sp2017-urban-sentiment/jsons/nytimes_restaurants.csv", index_col=None)

    i = 0
    nyt_id_to_review = {}
    nbhd_to_review = defaultdict(list)
    nyt_id_to_neighborhood = {}
    for row in data_frame.itertuples():
        review = getattr(row, 'summary')
        if isinstance(review, basestring):
            review = UnicodeDammit(review).unicode_markup.encode("utf-8")
            neighborhood = getattr(row, 'neighborhood').lower()
            review_id = str(int(getattr(row, 'review_id')))
            nyt_id_to_review[review_id] = review
            nbhd_to_review[neighborhood].append(review_id)
            nyt_id_to_neighborhood[review_id] = neighborhood.lower()

        if(i % 1000 == 0):
            print(str(i)+" completed.")
        i += 1

    return nyt_id_to_review, nbhd_to_review, nyt_id_to_neighborhood


def create_tfidf_matrix(doc_dict):
    print("Create TFIDF Matrix Function")
    listings = [(lid, text) for lid, text in doc_dict.iteritems()]
    listings_text = [listing[1] for listing in listings]
    #return listings_text

    listings_enumerate = list(enumerate(listings))

    print("Creating Vectorizers and vocabulary (word to index)")
    tfidf_vect = TfidfVectorizer(max_df=0.95, min_df=0.02, max_features=5000, stop_words='english', norm='l2', ngram_range=(1,1))
    tfidf_matrix = tfidf_vect.fit_transform(listings_text)

    tfidf_matrix_transpose=tfidf_matrix.transpose()
    words_compressed, sigma, docs_compressed = svds(tfidf_matrix_transpose, k=50) #word_compressed can convert query tfidf to svd form, docs_compressed is the compressed tfidf, put both on aws
    word_to_index = tfidf_vect.vocabulary_ #need to put on aws to convert query to idf form
    words_compressed = normalize(words_compressed, axis=1)
    docs_compressed=docs_compressed.transpose()


    print("Creating Dictionaries")
    # create idx_to_id & id_to_idx
    id_to_idx = {listings[item[0]][0]: item[0] for item in listings_enumerate}
    idx_to_id = {item[0]: listings[item[0]][0] for item in listings_enumerate}

    return docs_compressed, words_compressed, word_to_index, id_to_idx, idx_to_id, sigma, tfidf_vect.idf_



if __name__ == '__main__':

    neighborhood_to_listing_ids = neighborhood_to_listing_ids()
    with open('neighborhood_to_listing_ids.json', 'w') as fp:
        json.dump(neighborhood_to_listing_ids, fp)

    airbnb_listing_id_to_listing = listing_id_to_listing()
    with open('airbnb_listing_id_to_listing.json', 'w') as fp:
        json.dump(airbnb_listing_id_to_listing, fp)

    tfidf_compressed, words_compressed, word_to_index, airbnb_id_to_idx, airbnb_idx_to_id, sigma,idf_values = create_tfidf_matrix(airbnb_listing_id_to_listing)
    with open('tfidf_matrix.json', 'w') as fp:
        json.dump(tfidf_matrix, fp)
    json.dump(tfidf_compressed, open('airbnb_tfidf_compressed.json', 'w'), cls=NumpyEncoder)
    json.dump(words_compressed, open('airbnb_words_compressed.json','w'), cls=NumpyEncoder)
    json.dump(word_to_index, open('airbnb_word_to_index.json','w'), cls=NumpyEncoder)
    json.dump(sigma, open('airbnb_sigma.json','w'), cls=NumpyEncoder)
    json.dump(idf_values, open('airbnb_idf_values.json','w'), cls=NumpyEncoder)
    with open('airbnb_idx_to_id.json', 'w') as fp:
        json.dump(airbnb_idx_to_id, fp)
    with open('airbnb_id_to_idx.json', 'w') as fp:
        json.dump(airbnb_id_to_idx, fp)

    # nyt_id_to_review, nbhd_to_review, nyt_id_to_neighborhood = nyt_id_to_review()
    # with open('nyt_id_to_review.json', 'w') as fp:
    #     json.dump(nyt_id_to_review, fp)
    # with open('nyt_nbhd_to_review.json', 'w') as fp:
    #     json.dump(nbhd_to_review, fp)
    # with open('nyt_id_to_neighborhood.json', 'w') as fp:
    #     json.dump(nyt_id_to_neighborhood, fp)

    # nyt_tfidf_compressed, nyt_words_compressed, nyt_word_to_index, nyt_id_to_idx, nyt_idx_to_id, sigma, nyt_idf_values = create_tfidf_matrix(nyt_id_to_review)
    # json.dump(nyt_tfidf_compressed, open('nytimes_tfidf_compressed.json', 'w'), cls=NumpyEncoder)
    # json.dump(nyt_words_compressed, open('nytimes_words_compressed.json','w'),cls=NumpyEncoder)
    # json.dump(nyt_word_to_index, open('nytimes_word_to_index.json','w'),cls=NumpyEncoder)
    # json.dump(sigma, open('nytimes_sigma.json','w'),cls=NumpyEncoder)
    # json.dump(nyt_idf_values, open('nytimes_idf_values.json','w'),cls=NumpyEncoder)
    # with open('nyt_idx_to_id.json', 'w') as fp:
    #     json.dump(nyt_idx_to_id, fp)
    # with open('nyt_id_to_idx.json', 'w') as fp:
    #     json.dump(nyt_id_to_idx, fp)
