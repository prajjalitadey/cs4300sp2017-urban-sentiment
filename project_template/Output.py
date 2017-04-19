from Matrixizer import Matrixizer

class Output:
    #code
    """
    self.Matrixizer: Gets the relevant shit from the Matrix
    """
    def __init__(self, matrix_input):
        self.matrix = Matrixizer(matrix_input)
    
    def getNaiveImpl(self, query):
        neighborhood_to_score = self.matrix.query(query);
        res = sorted(neighborhood_to_score, key=neighborhood_to_score.__getitem__, reverse=True)
        return [(neighborhood, neighborhood_to_score[neighborhood]) for neighborhood in res]
        
        