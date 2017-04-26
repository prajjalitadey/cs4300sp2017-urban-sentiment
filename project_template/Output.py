from Matrixizer import Matrixizer
from Parser import Parser
import numpy as np

from collections import defaultdict


class Output:
    # Code
    """
    self.Matrixizer: Gets the relevant shit from the Matrix
    """
    def __init__(self, airbnb_input, nytimes_input):
        self.airbnb_matrix = Matrixizer(airbnb_input)
        self.nytimes_matrix = Matrixizer(nytimes_input)

    # Get the combined score of all criteria in the query
    def getCombinedScores(self, criteria_dict):
        all_airbnb = defaultdict(list)
        all_nytimes = defaultdict(list)

        for key in criteria_dict.keys():
            #Handle Airbnb scores
            airbnb_scores = criteria_dict[key]["airbnb_scores"]
            # airbnb scores is a list of score_dicts
            for score_dict in airbnb_scores:
                for neighborhood, score in score_dict.iteritems():
                    all_airbnb[neighborhood].append(score)

            #Handle NYTimes scores.
            nytimes_scores = criteria_dict[key]["nytimes_scores"]
            for score_dict in nytimes_scores:
                for neighborhood, score in score_dict.iteritems():
                    all_nytimes[neighborhood].append(score)

        # take mean of airbnb scores and of nytimes scores
        combined_airbnb = {}
        combined_nytimes = {}

        for neighborhood, scores in all_airbnb.iteritems():
                combined_airbnb[neighborhood.lower()] = np.mean(scores)
        for neighborhood, scores in all_nytimes.iteritems():
                combined_nytimes[neighborhood.lower()] = np.mean(scores)

        #Combine airbnb and nytimes scores
        combined_results = {}
        for neighborhood in combined_airbnb.keys():
            neighborhood = neighborhood.lower()
            if(neighborhood in combined_nytimes.keys()):
                combined_results[neighborhood] = np.mean([combined_airbnb[neighborhood], combined_nytimes[neighborhood]])
            else:
                combined_results[neighborhood] = combined_airbnb[neighborhood]

        return combined_results

    def getNaiveImpl(self, query):
        query_criteria = query.split(",")
        query_criteria  = [ query.strip() for query in query_criteria]
        criteria_results = {}

        for criteria in query_criteria:
            criteria_results[criteria] = {"airbnb_scores": [], "nytimes_scores": []}
            airbnb_scores = self.airbnb_matrix.query(criteria)
            nytimes_scores = self.nytimes_matrix.query(criteria)
            criteria_results[criteria]["airbnb_scores"].append(airbnb_scores)
            criteria_results[criteria]["nytimes_scores"].append(nytimes_scores)
            
        combined_results = self.getCombinedScores(criteria_results)
        return_me = {}
        print('hello its me')
        print(criteria_results[criteria]['airbnb_scores'])
        return_me = defaultdict(dict)
        for criteria in query_criteria:
            for neighborhood in criteria_results[criteria]['airbnb_scores'][0].keys():
                if neighborhood in criteria_results[criteria]['nytimes_scores'][0].keys():
                    return_me[criteria][neighborhood] = np.mean(criteria_results[criteria]['airbnb_scores'][0][neighborhood],
                                                                      criteria_results[criteria]['nytimes_scores'][0][neighborhood])
                else:
                    return_me[criteria][neighborhood] = criteria_results[criteria]['airbnb_scores'][0][neighborhood]        

        res = sorted(combined_results, key=combined_results.__getitem__, reverse=True)
        
        
        now_return_me = {}
        now_return_me['All'] = [(neighborhood, return_me[criteria] [neighborhood]) for neighborhood in res]
        now_return_me[query] = []
        
        for criteria in query_criteria:
            res = sorted(return_me[criteria], key=return_me[criteria].__getitem__, reverse=True)
            now_return_me[criteria] =  [(neighborhood, return_me[criteria] [neighborhood]) for neighborhood in res]
            
        print now_return_me
        
        return now_return_me