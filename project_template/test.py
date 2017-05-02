from .models import Docs
import Levenshtein
import json
from Parser import Parser
from Output import Output
from CribHub import CribHub

inst = CribHub()

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


def original_query(q):
    d = inst.handle_query(q)
    jsonarray = json.dumps(d)
    print("DONE WITH ORIGINAL QUERY")
    print(len(jsonarray))
    print("------------------------")
    return jsonarray


def requery(q):
    d = inst.rocchio(q)
    jsonarray = json.dumps(d)
    return jsonarray
