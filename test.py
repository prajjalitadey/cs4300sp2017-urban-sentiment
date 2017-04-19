import nltk
from PyDictionary import PyDictionary
from nltk.corpus import wordnet as wn

if __name__ == "__main__":
    # for i,j in enumerate(wn.synsets('happy')):
    #   print j.lemma_names

    dictionary = PyDictionary()

    # print(dictionary.printMeanings())
    # print(dictionary.getMeanings())
    print (dictionary.synonym("filthy"))

    # print (dictionary.translateTo("hi"))
