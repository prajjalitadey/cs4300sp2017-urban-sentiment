import urllib2, json
import pandas as pd
from collections import defaultdict
from bs4 import UnicodeDammit

##### LOAD DATA FROM S3 #####
def load_data():

    file1 = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_dict.json')
    airbnb_dictionary = json.load(file1, encoding='utf8') 

    return { "airbnb_dictionary" : airbnb_dictionary }

    
def neighborhood_to_listing_ids():

    data_frame = pd.DataFrame.from_csv("/Users/lyllayounes/Desktop/cs4300sp2017-urban-sentiment/jsons/nyc_combination.csv", index_col=None)
    #data_frame = pd.DataFrame.from_csv("/Users/lyllayounes/Desktop/cs4300sp2017-urban-sentiment/jsons/test.csv", index_col=None)

    airbnb_dict = defaultdict(dict)

    i = 0

    neighborhood_to_listing_ids = defaultdict(list)

    for row in data_frame.itertuples():

        review = getattr(row, 'comments')

        if isinstance(review, basestring):
            review = UnicodeDammit(review).unicode_markup.encode("utf-8")
            neighborhood = (getattr(row, 'neighbourhood')).lower()
            listing_id = getattr(row, 'listing_id')

            neighborhood_to_listing_ids[neighborhood].append(listing_id)

        if(i%1000==0):
            print(str(i)+" completed.")

        i+=1

    return neighborhood_to_listing_ids



def listing_id_to_listing():

    data_frame = pd.DataFrame.from_csv("/Users/lyllayounes/Desktop/cs4300sp2017-urban-sentiment/jsons/nyc_combination.csv", index_col=None)
    #data_frame = pd.DataFrame.from_csv("/Users/lyllayounes/Desktop/cs4300sp2017-urban-sentiment/jsons/test.csv", index_col=None)

    i = 0

    listing_id_to_listing_temp = defaultdict(list)

    for row in data_frame.itertuples():

        review = getattr(row, 'comments')

        if isinstance(review, basestring):
            review = UnicodeDammit(review).unicode_markup.encode("utf-8")
            neighborhood = (getattr(row, 'neighbourhood')).lower()
            listing_id = getattr(row, 'listing_id')

            listing_id_to_listing_temp[listing_id].append(review)

        if(i%1000==0):
            print(str(i)+" completed.")

        i+=1

    # for listing_id in listing_id_to_listing.keys():
    listing_id_to_listing = { lid: ', '.join(str_list) for lid, str_list in listing_id_to_listing_temp.iteritems()}


    return listing_id_to_listing




if __name__ == '__main__':

    # neighborhood_to_listing_ids = parse_data()
    # with open('neighborhood_to_listing_ids.json', 'w') as fp:
    #     json.dump(neighborhood_to_listing_ids, fp)

    listing_id_to_listing = listing_id_to_listing()
    with open('listing_id_to_listing.json', 'w') as fp:
        json.dump(listing_id_to_listing, fp)












