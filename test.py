'''
import collections
import pymysql
import numpy as np
import os
import re
import comment_scoring
'''
class Stream(object):
    def __init__(self, preprocess_fn, dictionary, stopwords):
        self.preprocess_fn = preprocess_fn
        self.dictionary = dictionary
        self.stopwords = stopwords

    def __iter__(self):
        with open('./sample_corpus.txt', 'r') as f:
            while True:
                text = f.readline().strip()
                if text == '':
                    break
                yield self.dictionary.doc2bow(
                    self.preprocess_fn(text, self.stopwords))

'''
stream = Stream(comment_scoring.preprocess)
dictionary = build_dictionary(stream)

for word_id, word in dictionary.items():
    print(word_id, word)

corpus = [dictionary.doc2bow(text) for text in stream]


for vec in corpus:
    print(vec)
'''
vals = [1,3,5,7,9]
print(sum(val for val in vals))





