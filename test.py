import numpy as np
from sklearn import preprocessing
import utilities
import run
from gensim import corpora

class Stream(object):
    def __init__(self, preprocess_fn, stopwords):
        self.preprocess_fn = preprocess_fn
        self.stopwords = stopwords
        self.d = {}

    def __iter__(self):
        with open('./sample_corpus.txt', 'r') as f:
            idx = 0
            while True:
                text = f.readline().strip()
                if text == '':
                    break
                self.d[text[0]] = idx
                idx += 1
                yield self.preprocess_fn(text, self.stopwords)



stopwords = utilities.load_stopwords('./stopwords.txt')

stream = Stream(utilities.preprocess, stopwords)

print(stream.d)

dictionary = corpora.Dictionary(stream)

print(stream.d)







