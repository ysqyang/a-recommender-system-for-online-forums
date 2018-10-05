import numpy as np
from sklearn import preprocessing
import utilities
import random
from gensim import corpora, models
import collections
import stream
import pandas as pd
from pprint import pprint
import time
from datetime import date, datetime
'''
class Stream(object):
    def __init__(self, topic_id, preprocess_fn, stopwords):
        self.topic_id = topic_id
        self.preprocess_fn = preprocess_fn
        self.stopwords = stopwords
        self.id_to_index = {}

    def __iter__(self):
        with open('./sample_corpus_{}.txt'.format(self.topic_id), 'r') as f:
            line_no = 0
            while True:
                text = f.readline().strip()
                if text == '':
                    break
                _id = int(text[:5]) 
                self.id_to_index[_id] = line_no
                line_no += 1
                yield self.preprocess_fn(text[5:], self.stopwords)

def get_scores(results, weights, id_to_index):
    s, scores, scaler = sum(weights), {}, preprocessing.MinMaxScaler()
    norm_weights = [wt/s for wt in weights]

    features_norm = scaler.fit_transform(np.array(results)[..., 1:])

    pprint(features_norm)
    for result, feature_vec in zip(results, features_norm):
        corpus_index = id_to_index[result[0]]
        scores[corpus_index] = np.dot(feature_vec, norm_weights)

    return scores

def get_word_weights(corpus_under_topic, dictionary, scores, alpha=0.7, smartirs='atn'):
    word_weight = collections.defaultdict(float)

    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_under_topic]
    language_model = models.TfidfModel(corpus_bow, smartirs=smartirs)    

    # get the max score under each topic for normalization purposes
    max_score = max(scores.values())
    print('max_score:', max_score)
    for i, doc in enumerate(corpus_bow):
        converted = language_model[doc]
        print(converted)
        max_word_weight = max([x[1] for x in converted])
        if i == 0:
            coeff, score_norm = alpha, 1
        else:
            coeff, score_norm = 1-alpha, scores[i]/max_score
        for word in converted:
            word_weight_norm = word[1]/max_word_weight
            word_weight[dictionary[word[0]]] += coeff*score_norm*word_weight_norm

    return word_weight

def get_word_weights_all(tid_to_table, results, weights, preprocess_fn, 
                         stopwords, alpha=0.7, smartirs='atn'):

    word_weight = {}

    # create a Corpus_under_topic object for each topic
    for topic_id in tid_to_table:
        corpus = Stream(topic_id, preprocess_fn, stopwords)
        
        dictionary = corpora.Dictionary(corpus)
        for _id, word in dictionary.items():
            print(_id, word)

        print(corpus.id_to_index)
        
        scores = get_scores(results[topic_id], weights, corpus.id_to_index)        
        
        pprint(scores)
        word_weight[topic_id] = get_word_weights(
                                 corpus, dictionary, scores, alpha, smartirs)

    return word_weight

def get_top_k_words(word_weight, k):
    if k > len(word_weight):
        k = len(word_weight)

    word_weight = [(w, word_weight[w]) for w in word_weight]
    
    word_weight.sort(key=lambda x:x[1], reverse=True)

    return [x[0] for x in word_weight[:k]] 

topics = [22, 14]
weights = [1,4,2,5]
stopwords = utilities.load_stopwords('./stopwords.txt')

results = {}
results[22] = [(95,)+tuple(random.randrange(10) for _ in range(4)),
              (173,)+tuple(random.randrange(10) for _ in range(4)),
              (315,)+tuple(random.randrange(10) for _ in range(4)), 
              (1004,)+tuple(random.randrange(10) for _ in range(4))
              ]

pprint(results[22])

results[14] = [(11,)+tuple(random.randrange(10) for _ in range(4)),
              (263,)+tuple(random.randrange(10) for _ in range(4)),
              (477,)+tuple(random.randrange(10) for _ in range(4)), 
              (985,)+tuple(random.randrange(10) for _ in range(4))
             ]

print(results[14])


word_weight = get_word_weights_all(topics, results, weights, utilities.preprocess, 
                                   stopwords)

print(word_weight)

profile_words = {tid:get_top_k_words(weight, 3)
                     for tid, weight in word_weight.items()}

pprint(profile_words)

'''
today = date.today()
print(today)


s = '8/17/2017'
past = datetime.strptime('8/17/2017', '%m/%d/%Y')

print(past.date())

print((today-past.date()).days)












