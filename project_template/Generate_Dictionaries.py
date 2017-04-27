import pandas as pd

from collections import defaultdict
from bs4 import UnicodeDammit


def generateAirbnbDictionary():
    data_frame = pd.DataFrame.from_csv("/Users/lyllayounes/Desktop/cs4300sp2017-urban-sentiment/jsons/test.csv", index_col=None)
    airbnb_dict = defaultdict(dict)

    for row in data_frame.itertuples():
        review = getattr(row, 'comments')
        neighborhood = getattr(row, 'neighbourhood')
        listing_id = getattr(row, 'listing_id')

        if(neighborhood in (airbnb_dict[listing_id]).keys()):
            airbnb_dict[listing_id]["reviews"].append(review) 
        else:
            airbnb_dict[listing_id] = {"neighborhood":neighborhood,"reviews":[review],"id":listing_id}

    return airbnb_dict

def generateNYTimesDictionary():
    data_frame = pd.DataFrame.from_csv("/Users/lyllayounes/Desktop/cs4300sp2017-urban-sentiment/jsons/nytimes_restaurants.csv", index_col=None)
    nyt_dict = defaultdict(dict)

    for row in data_frame.itertuples():
        review = getattr(row, 'summary')
        neighborhood = getattr(row, 'neighborhood')
        review_id = getattr(row, 'review_id')
        name = getattr(row, 'name')

        
        nyt_dict[review_id] = {"neighborhood":neighborhood,"reviews":[review],"id":review_id,"restaurant_name":name}

    return nyt_dict
    



if __name__ == '__main__':
    generateAirbnbDictionary()
    generateNYTimesDictionary()
