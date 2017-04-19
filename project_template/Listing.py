class Listing:

    """
    self.id: id of this house
    self.reviews:  the reviews of this listing
    self.score: score for this listing
    self.neighborhood: which neighborhood this listing is in
    """

    def  __init__(self, listing_id, listing_reviews, neighborhood):
        self.id = listing_id
        self.reviews = listing_reviews
        self.neighborhood = neighborhood
        self.score = 0

    def getID(self):
        return self.id

    def getReviews(self):
        return self.reviews

    def getNeighborhood(self):
        return self.neighborhood

    def setScore(self, score):
        self.score = score

    def getScore(self):
        return self.score
