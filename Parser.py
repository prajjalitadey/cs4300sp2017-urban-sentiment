import csv

class Parser:

    def __init__(self):
        pass

    def parseAirbnb(self):
        # open reviews file & create reviews dictionary
        reviewsfile = open('jsons/nycreviews.csv', 'rb')
        rvreader = csv.reader(reviewsfile)
        reviews = defaultdict(list)
        for row in rvreader:
            listing_id = row[0]
            comment = row[5]
            reviews[listing_id].append(comment)

        # open listings file & create listings dictionary
        listingsfile = open('jsons/nyclistings.csv', 'rb')
        listingsreader = csv.reader(listingsfile)
        listings = defaultdict(list)
        for row in listingsreader:
            if row[5] != "neighbourhood":
                listing_id = row[0]
                neighborhood = row[5]
                listings[neighborhood].append(listing_id)

        # create listing_reviews dictionary that combines both listings and reviews
        listing_reviews = {}
        for neighborhood,listing_ids in listings.iteritems():
            nested_list = [reviews[listing] for listing in listing_ids if listing in reviews.keys()]
            listings_reviews[neighborhood] = [item for sublist in nested_list for item in sublist]


        # Combining list of reviews for each neighborhood, into a single string
        # Also converting all the reviews text into UTF-8 encoding, to build the tf-idf later
        for neighborhood in listings_reviews:
            if(len(listings_reviews[neighborhood])>1):
                k=' '.join(listings_reviews[neighborhood])
                m=UnicodeDammit(k)
                s=m.unicode_markup
                listings_reviews[neighborhood]=s.encode("utf-8")
            else:
                if(len(listings_reviews[neighborhood])==1):
                    listings_reviews[neighborhood]=listings_reviews[neighborhood][0]
                if(len(listings_reviews[neighborhood])==0):
                    listings_reviews[neighborhood]='the' # i think empty text creates an error, hence this
        self.listings_reviews = listing_reviews

    def parseTwitter(self):
        pass

    def getReviews(self):
        return self.listings_reviews

    def __unicode__(self):
        return unicode(self.some_field) or u''
