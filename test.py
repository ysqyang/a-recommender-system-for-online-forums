import collections
import pymysql
import numpy as np
import os
import re
import comment_scoring
from gensim import corpora, models, similarities
from sklearn import preprocessing
import pickle

db_info = ('192.168.1.102','tgbweb','tgb123321','taoguba',3307)

STOPWORDS = './stopwords.txt'

def build_dictionary(corpus_stream):
    return corpora.Dictionary(corpus_stream) 


class Stream(object):
    def __init__(self, preprocess_fn):
        self.preprocess_fn = preprocess_fn

    def __iter__(self):
        for i in range(4):
            with open('./corpus_{}.txt'.format(i)) as f:
                while True:
                    text = f.readline().strip()
                    if text == '':
                        break
                    yield self.preprocess_fn(text, STOPWORDS)

class Stream_num(object):
    def __init__(self, vals):
        self.vals = vals

    def __iter__(self):
        for val in self.vals:
            yield val


'''
stream = Stream(comment_scoring.preprocess)
dictionary = build_dictionary(stream)

for word_id, word in dictionary.items():
    print(word_id, word)

corpus = [dictionary.doc2bow(text) for text in stream]


for vec in corpus:
    print(vec)
'''

i = 0
if i:
    print('great')
else:
    print('bad')




