import csv
import pandas as pd

from collections import defaultdict
from bs4 import UnicodeDammit

from Listing.py import Listing


class Parser:

    def __init__(self):
        self.listings_reviews = {}

    def parseAirbnb(self):
        # use pandas to combine both csvs
        df_reviews = pd.DataFrame.from_csv('jsons/nycreviews.csv', index_col=None)
        df_listings = pd.DataFrame.from_csv('jsons/nyclistings.csv', index_col=None)
        df_reviews = df_reviews[['listing_id', 'comments']]
        df_listings = df_listings[['id', 'neighbourhood']]
        df_listings = df_listings.rename(columns={'id': 'listing_id'})
        df_combination = pd.merge(df_listings, df_reviews, on='listing_id')
        df_combination.to_csv('jsons/nyc_combination.csv')

        reviews = defaultdict(list)
        listings = defaultdict(list)
        for row in df_combination.itertuples():
            reviews[getattr(row, 'listing_id')].append(getattr(row, 'comments'))
            listings[getattr(row, 'neighbourhood')].append(getattr(row, 'listing_id'))

        # create neighborhoods dictionary with listing objects
        neighborhoods = {}
        for neighborhood, listing_ids in listings.iteritems():
            all_list_objs = [Listing(lid, UnicodeDammit(' '.join(reviews[lid])).unicode_markup.encode("utf-8"), neighborhood) for lid in listing_ids if lid in reviews.keys()]
            neighborhoods[neighborhood] = all_list_objs

        # change encoding of text
        # UnicodeDammit(text_ex).unicode_markup.encode("utf-8")
        # check with df_test[2515][147]

        # Combining list of reviews for each neighborhood, into a single string
        # Also converting all the reviews text into UTF-8 encoding, to build the tf-idf later
        # for neighborhood in listings_reviews:
        #     if(len(listings_reviews[neighborhood]) > 1):
        #         k = ' '.join(listings_reviews[neighborhood])
        #         m = UnicodeDammit(k)
        #         s = m.unicode_markup
        #         listings_reviews[neighborhood] = s.encode("utf-8")
        #     else:
        #         if (len(listings_reviews[neighborhood]) == 1):
        #             listings_reviews[neighborhood] = listings_reviews[neighborhood][0]
        #         if (len(listings_reviews[neighborhood]) == 0):
        #             listings_reviews[neighborhood] = 'the'  # i think empty text creates an error, hence this

        self.listings_reviews = neighborhoods

    def parseNYT(self):
        pass

    def getReviews(self):
        return self.listings_reviews

    def __unicode__(self):
        return unicode(self.some_field) or u''
