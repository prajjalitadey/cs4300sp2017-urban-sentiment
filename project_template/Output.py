from Matrixizer import Matrixizer
import numpy as np

class Output:
    # Code
    """
    self.Matrixizer: Gets the relevant shit from the Matrix
    """
    def __init__(self, airbnb_input, nytimes_input):
        self.airbnb_matrix = Matrixizer(airbnb_input)
        self.nytimes_matrix = Matrixizer(nytimes_input)
    
    def getNaiveImpl(self, query):
        query_criteria = query.split(",")
        criteria_results[criteria] = {"airbnb_scores":[],"nytimes_scores":[]}
        for criteria in query_criteria:
            airbnb_scores = self.airbnb_matrix.query(criteria)
            nytimes_scores = self.nytimes_matrix.query(criteria)
            criteria_results["airbnb_scores"].append(airbnb_scores)
            criteria_results["nytimes_scores"].append(nytimes_scores)

        combined_results = getCombinedScores(criteria_results)
        
        res = sorted(combined_results, key=combined_results.__getitem__, reverse=True)
        return [(neighborhood, combined_results[neighborhood]) for neighborhood in res]
            

    # Get the combined score of all criteria in the query
    def getCombinedScores(self,criteria_dict):
        
        #Handle Airbnb scores.
        airbnb_scores = criteria_dict["airbnb_scores"]
        neighborhoods = airbnb_scores[0].keys()
        combined_airbnb = defaultdict(list)
        for score_dict in airbnb_scores:
            for neighborhood, score in score_dict.itervalues():
                combined_airbnb[neighborhood].append(score)

        for neighborhood, scores in combined_airbnb.itervalues():
            combined_airbnb[neighborhood] = np.mean(scores)

        #Handle NYTimes scores.
        nytimes_scores = criteria_dict["nytimes_scores"]
        neighborhoods = nytimes_scores[0].keys()
        combined_nytimes = defaultdict(list)
        for score_dict in nytimes_scores:
            for neighborhood, score in score_dict.itervalues():
                combined_nytimes[neighborhood].append(score)

        for neighborhood, scores in combined_nytimes.itervalues():
            combined_nytimes[neighborhood] = np.mean(scores)


        #Combine results. 
        combined_results = {}
        for neighborhood in combined_airbnb.keys():
            if(neighborhood in combined_nytimes.keys()):
                combined_results[neighborhood] = np.mean([combined_airbnb[neighborhood],combined_nytimes[neighborhood]])

        return combined_results




            