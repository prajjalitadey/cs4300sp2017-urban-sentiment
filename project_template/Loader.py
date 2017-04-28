import urllib2
import json
import pandas as pd
from collections import defaultdict
from bs4 import UnicodeDammit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


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
    return neighborhood_to_listing_ids


# airbnb function
def listing_id_to_listing():
    print("Listing_it_to_listing Function")
    data_frame = pd.DataFrame.from_csv("/Users/prajjalitadey/Documents/Spring2017/4300/cs4300sp2017-urban-sentiment/jsons/nyc_combination.csv", index_col=None)
    # data_frame = pd.DataFrame.from_csv("/Users/prajjalitadey/Documents/Spring2017/4300/cs4300sp2017-urban-sentiment/jsons/test.csv", index_col=None)

    i = 0
    listing_id_to_listing_temp = defaultdict(list)
    for row in data_frame.itertuples():
        review = getattr(row, 'comments')
        if isinstance(review, basestring):
            review = UnicodeDammit(review).unicode_markup.encode("utf-8")
            neighborhood = (getattr(row, 'neighbourhood')).lower()
            listing_id = getattr(row, 'listing_id')

            listing_id_to_listing_temp[listing_id].append(review)

        if(i % 1000 == 0):
            print(str(i)+" completed.")
        i += 1

    # listing_id: a string that is the combination of all reviews for that listing
    listing_id_to_listing = {lid: ', '.join(str_list) for lid, str_list in listing_id_to_listing_temp.iteritems()}

    return listing_id_to_listing


# nyt function
def nyt_id_to_review():
    print("NYT_ID_to_Review Function")
    data_frame = pd.DataFrame.from_csv("/Users/prajjalitadey/Documents/Spring2017/4300/cs4300sp2017-urban-sentiment/jsons/nytimes_restaurants.csv", index_col=None)

    i = 0
    nyt_id_to_review = {}
    nbhd_to_review = defaultdict(list)
    for row in data_frame.itertuples():
        review = getattr(row, 'summary')
        if isinstance(review, basestring):
            review = UnicodeDammit(review).unicode_markup.encode("utf-8")
            # name = getattr(row, 'name')
            neighborhood = getattr(row, 'neighborhood')
            review_id = getattr(row, 'review_id')
            nyt_id_to_review[review_id] = review
            nbhd_to_review[neighborhood].append(review_id)
        if(i % 1000 == 0):
            print(str(i)+" completed.")
        i += 1

    return nyt_id_to_review, nbhd_to_review


def create_tfidf_matrix(doc_dict):
    print("Create TFIDF Matrix Function")
    listings = [(lid, text) for lid, text in doc_dict.iteritems()]
    listings_text = [listing[1] for listing in listings]
    listings_enumerate = list(enumerate(listings))

    print("Creating Vectorizers")
    tfidf_vect = TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=5000, stop_words='english', norm='l2')
    tfidf_matrix = tfidf_vect.fit_transform(listings_text)
    tfidf_svd_vect = TruncatedSVD(n_components=500, random_state=42)
    tfidf_svd_matrix = tfidf_svd_vect.fit_transform(tfidf_matrix).tolist()

    print("Creating Dictionaries")
    # create idx_to_id & id_to_idx
    id_to_idx = {listings[item[0]][0]: item[0] for item in listings_enumerate}
    idx_to_id = {item[0]: listings[item[0]][0] for item in listings_enumerate}

    return tfidf_svd_matrix, idx_to_id, id_to_idx


if __name__ == '__main__':

    # neighborhood_to_listing_ids = parse_data()
    # with open('neighborhood_to_listing_ids.json', 'w') as fp:
    #     json.dump(neighborhood_to_listing_ids, fp)

    airbnb_listing_id_to_listing = listing_id_to_listing()
    # with open('airbnb_listing_id_to_listing.json', 'w') as fp:
    #     json.dump(airbnb_listing_id_to_listing, fp)

    airbnb_tfidf, airbnb_idx_to_id, airbnb_id_to_idx = create_tfidf_matrix(airbnb_listing_id_to_listing)
    with open('airbnb_tfidf.json', 'w') as fp:
        json.dump(airbnb_tfidf, fp)
    with open('airbnb_idx_to_id.json', 'w') as fp:
        json.dump(airbnb_idx_to_id, fp)
    with open('airbnb_id_to_idx.json', 'w') as fp:
        json.dump(airbnb_id_to_idx, fp)

    nyt_id_to_review, nbhd_to_review = nyt_id_to_review()
    with open('nyt_id_to_review.json', 'w') as fp:
        json.dump(nyt_id_to_review, fp)
    with open('nbhd_to_review.json', 'w') as fp:
        json.dump(nbhd_to_review, fp)

    nyt_tfidf, nyt_idx_to_id, nyt_id_to_idx = create_tfidf_matrix(nyt_id_to_review)
    with open('nyt_tfidf.json', 'w') as fp:
        json.dump(nyt_tfidf, fp)
    with open('nyt_idx_to_id.json', 'w') as fp:
        json.dump(nyt_idx_to_id, fp)
    with open('nyt_id_to_idx.json', 'w') as fp:
        json.dump(nyt_id_to_idx, fp)
