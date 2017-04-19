from .models import Docs
import os
import Levenshtein
import json
from Parser import Parser
from Output import Output
from Listing import Listing
from nltk.corpus import wordnet as wn


def read_file(n):
	path = Docs.objects.get(id = n).address;
	file = open(path)
	transcripts = json.load(file)
	return transcripts

def _edit(query, msg):
    return Levenshtein.distance(query.lower(), msg.lower())

def find_similar(q):
	transcripts = read_file(1)
	result = []
	for transcript in transcripts:
		for item in transcript:
			m = item['text']
			result.append(((_edit(q, m)), m))

	return sorted(result, key=lambda tup: tup[0])

def get_neighborhood_ranking(q):
	# listing = {}
	# listing['Wall Street'] = [Listing(1, 'great place close to wall street', 'Wall Street')]
	# listing['Fordham'] = [Listing(2, 'shitty ass neighborhood', 'Fordham')]
	# listing['Eastchester'] = [Listing(3, 'Westchester', 'Eastchester')]
	# listing['Manhanttan'] = [Listing(4, 'yes I know its a borough', 'Manhattan')]

	#transcripts = read_file(11)
	query = "new york, outside of manhattan, quiet area, elderly people"
	parser = Parser()
	#parser.parseAirbnb("/Users/lyllayounes/Desktop/cs4300sp2017-urban-sentiment/jsons/test.csv")
	parser.parseAirbnb("./jsons/test.csv")
	listing = parser.getReviews()
	output = Output(listing)
	tups = output.getNaiveImpl(query)
	tmp = []
	tmp.append(["Query: "+query])
	tmp.append(["-------------------------------------------------"])

	for item in tups:
		tmp.append(item)

	return tmp
