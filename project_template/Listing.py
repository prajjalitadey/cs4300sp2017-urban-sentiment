class Review:
    """
    "parent" class
    """

    def __init__(self, neighborhood, review):
        # self.id = doc_id
        self.review = review
        self.neighborhood = neighborhood
        self.score = 0

    def getReviews(self):
        return self.review

    def getNeighborhood(self):
        return self.neighborhood

    def setScore(self, score):
        self.score = score

    def getScore(self):
        return self.score


class Listing(Review):

    def __init__(self, neighborhood, review, doc_id):
        Review.__init__(self, neighborhood, review)
        self.id = doc_id

    def getID(self):
        return self.id


class Restaurant(Review):

    def __init__(self, neighborhood, review, name):
        Review.__init__(self, neighborhood, review)
        self.restaurant_name = name

    def getName(self):
        return self.restaurant_name
