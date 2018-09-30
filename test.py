import collections
import pymysql
import numpy as np
import os
import re
import comment_scoring
from gensim import corpora, models, similarities

db_info = ('192.168.1.102','tgbweb','tgb123321','taoguba',3307)

class Corpus_stream(object):
    '''
    Corpus object for streaming preprocessed texts
    '''
    def __init__(self, corpus_path, stopwords_path, preprocess_fn, dictionary):
        self.corpus_path = corpus_path
        self.stopwords_path = stopwords_path
        self.preprocess_fn = preprocess_fn
        self.dictionary = dictionary

    def __next__(self):
        with open(self.corpus_path, 'r') as f:
            raw_text = f.readline().strip()
        
        return self.dictionary.doc2bow(
             self.preprocess_fn(raw_text, self.stopwords_path))

    def __iter__(self):
        with open(self.corpus_path, 'r') as f:
            while True:
                raw_text = f.readline().strip()
                if raw_text == '':
                    break
                yield self.dictionary.doc2bow(
                    self.preprocess_fn(raw_text, self.stopwords_path))


def build_dictionary(corpus_path, preprocess_fn, stopwords_path):
    '''
    Builds a dictionary from a processed corpus
    Args:
    corpus_path:    path for raw corpus file
    preprocess_fn:  function to preprocess raw text
    stopwords_path: stopword file path
    '''
    return corpora.Dictionary(preprocess_fn(line.rstrip(), stopwords_path) 
                              for line in open(corpus_path, 'r')) 

i = 0

a, b = 1,2 if i == 0 
       else 3, 4