import pandas as pd

from collections import defaultdict
from bs4 import UnicodeDammit

from Listing import Listing, Restaurant


class Parser:

    def __init__(self):
        self.airbnb_reviews = {}
        self.nyt_reviews = {}

    def parseAirbnb(self, csvfile):
        df_combination = pd.DataFrame.from_csv(csvfile, index_col=None)

        reviews = defaultdict(list)
        listings = defaultdict(list)
        for row in df_combination.itertuples():
            text = getattr(row, 'comments')
            if isinstance(text, basestring):
                reviews[getattr(row, 'listing_id')].append(UnicodeDammit(text).unicode_markup.encode("utf-8"))
                listings[getattr(row, 'neighbourhood')].append(getattr(row, 'listing_id'))

        # create reviews dictionary with listing objects
        reviews = {}
        for neighborhood, listing_ids in listings.iteritems():
            all_list_objs = [Listing(lid, ' '.join(reviews[lid]), neighborhood) for lid in listing_ids if lid in reviews.keys()]
            reviews[neighborhood] = all_list_objs

        self.airbnb_reviews = reviews

    def parseNYTimes(self):
        # TODO
        df = pd.DataFrame.from_csv('jsons/nytimes_restaurants.csv', index_col=None)

        reviews = defaultdict(list)
        for row in df.itertuples():
            nbhd = getattr(row, 'neighborhood')
            review = getattr(row, 'summary')
            if isinstance(nbhd, basestring) and isinstance(review, basestring):
                review = UnicodeDammit(review).unicode_markup.encode("utf-8")  # fix encoding
                reviews[nbhd].append(Restaurant(getattr(row, 'reviewid'), nbhd.lower(), review, getattr(row, 'name')))

        self.nyt_reviews = reviews

    def getAirbnbReviews(self):
        return self.airbnb_reviews

    def getNYTimesReviews(self):
        return self.nyt_reviews

    def __unicode__(self):
        return unicode(self.some_field) or u''
