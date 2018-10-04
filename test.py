import numpy as np
from sklearn import preprocessing
import utilities
import random
from gensim import corpora, models
import collections
import stream
import pandas as pd
from pprint import pprint

class Stream(object):
    def __init__(self, preprocess_fn, stopwords):
        self.preprocess_fn = preprocess_fn
        self.stopwords = stopwords

    def __iter__(self):
        with open('./sample_corpus.txt', 'r') as f:
            while True:
                text = f.readline().strip()
                if text == '':
                    break
                yield self.preprocess_fn(text, self.stopwords)

def get_scores(results, weights, id_to_index):
    s, scores, scaler = sum(weights), {}, preprocessing.MinMaxScaler()
    norm_weights = [wt/s for wt in weights]

    features_norm = scaler.fit_transform(np.array(results)[..., 1:])

    pprint(features_norm)
    for result, feature_vec in zip(results, features_norm):
        corpus_index = id_to_index[result[0]]
        scores[corpus_index] = np.dot(feature_vec, norm_weights)

    return scores

def get_word_weights(corpus_under_topic, dictionary, topic_id, model, 
                 normalize, scores, alpha=0.7):
    word_weight = collections.defaultdict(float)

    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_under_topic]
    language_model = model(corpus_bow, normalize=normalize)    

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
            word_weight[word[0]] += coeff*score_norm*word_weight_norm

    return word_weight

def get_word_weights_all(db, tid_to_table, features, weights, preprocess_fn, 
                         stopwords, normalize, alpha):

    word_weight = {}

    # create a Corpus_under_topic object for each topic
    for topic_id in tid_to_table:
        corpus = stream.Corpus_under_topic(db, topic_id, 
                                           tid_to_table[topic_id], 
                                           preprocess_fn, stopwords)
        
        dictionary = corpora.Dictionary(corpus)
        
        scores = get_scores(db, topic_id, features, weights, 
                            corpus.reply_id_to_corpus_index)        
        
        word_weight[topic_id] = get_word_weights(
                                 corpus, dictionary, topic_id, 
                                 model, normalize, scores, alpha)

    return word_weight

def get_top_k_words(word_weight, k):
    if k > len(word_weight):
        k = len(word_weight)

    word_weight = [(w, word_weight[w]) for w in word_weight]
    
    word_weight.sort(key=lambda x:x[1], reverse=True)

    return [x[0] for x in word_weight[:k]] 


id_to_index = {22: 0, 315: 1, 173: 2, 1004: 3, 95: 4}
results = [(22,)+tuple([random.randrange(10) for _ in range(4)]),
           (315,)+tuple(random.randrange(10) for _ in range(4)),
           (173,)+tuple(random.randrange(10) for _ in range(4)),
           (1004,)+tuple(random.randrange(10) for _ in range(4)),
           (95,)+tuple(random.randrange(10) for _ in range(4))]

weights = [1,4,2,5]
scores = get_scores(results, weights, id_to_index)

print(scores)

stopwords = utilities.load_stopwords('./stopwords.txt')
stream = Stream(utilities.preprocess, stopwords)
topic_id = 3
dictionary = corpora.Dictionary(stream)

print(dictionary.token2id)
word_weights = get_word_weights(stream, dictionary, topic_id, models.TfidfModel, 
                 False, scores, alpha=0.7)

print(word_weights)













