import pandas as pd

from collections import defaultdict
from bs4 import UnicodeDammit

from Listing import Listing


class Parser:

    def __init__(self):
        self.listings_reviews = {}

    def parseAirbnb(self, csvfile):
        df_combination = pd.DataFrame.from_csv(csvfile, index_col=None)

        reviews = defaultdict(list)
        listings = defaultdict(list)
        for row in df_combination.itertuples():
            text = getattr(row, 'comments')
            if isinstance(text, basestring):
                reviews[getattr(row, 'listing_id')].append(UnicodeDammit(text).unicode_markup.encode("utf-8"))
                listings[getattr(row, 'neighbourhood')].append(getattr(row, 'listing_id'))

        # create neighborhoods dictionary with listing objects
        neighborhoods = {}
        for neighborhood, listing_ids in listings.iteritems():
            all_list_objs = [Listing(lid, ' '.join(reviews[lid]), neighborhood) for lid in listing_ids if lid in reviews.keys()]
            neighborhoods[neighborhood] = all_list_objs

        # change encoding of text
        # UnicodeDammit(text).unicode_markup.encode("utf-8")
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
