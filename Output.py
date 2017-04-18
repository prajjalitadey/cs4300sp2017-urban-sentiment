class Output:
    #code
    """
    self.Matrixizer: Gets the relevant shit form the Matrix
    """
    def __init__(self):
        self.matrix = Matrixizer()
    
    def getRelevantData(self):
        #we can get data from multiple machine learning techniques and return it as tuples from here
        return self.matrix.getTuples();
        
        