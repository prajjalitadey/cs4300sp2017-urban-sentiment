from .models import Docs
import Levenshtein
import json
from Parser import Parser
from Output import Output


def read_file(n):
    path = Docs.objects.get(id=n).address
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
    query = q

    parser = Parser()
    parser.parseAirbnb("./jsons/test.csv")
    parser.parseNYTimes("./jsons/nytimes_restaurants.csv")
    airbnb = parser.getAirbnbReviews()
    nytimes = parser.getNYTimesReviews()

    output = Output(airbnb, nytimes)
    results = output.getNaiveImpl(query)

    d = results
    jsonarray = json.dumps(d)

    return jsonarray
