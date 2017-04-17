import csv

class Parser:
    
    def getcsvs(self):
        reviewsfile = open('jsons/nycreviews.csv', 'rb')
        rvreader = csv.reader(reviewsfile)
        reviews = defaultdict(list)
        for row in rvreader:
            listing_id = row[0]
            comment = row[5]
            reviews[listing_id].append(comment)
       
        listingsfile = open('jsons/nyclistings.csv', 'rb')
        listingsreader = csv.reader(listingsfile)
        listings = defaultdict(list)
        for row in listingsreader:
            if row[5] != "neighbourhood":
                listing_id = row[0]
                neighborhood = row[5]
                listings[neighborhood].append(listing_id)
        return reviews, listings
    
    def parsecsvs(self, reviews, listings):
        "returns a dictionary with the neighborhood listings as a string"
        for neighborhood,listing_ids in listings.iteritems():
            nested_list = [reviews[listing] for listing in listing_ids if listing in reviews.keys()]
            listings_reviews[neighborhood] = [item for sublist in nested_list for item in sublist]
        
        #Combining list of reviews for each neighborhood, into a single string
        #Also converting all the reviews text into UTF-8 encoding, to build the tf-idf later
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
        return listings_reviews
    
    def __unicode__(self):
        return unicode(self.some_field) or u''
                    