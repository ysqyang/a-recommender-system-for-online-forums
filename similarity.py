from gensim import corpora
import utils
import copy
import numpy as np
from pprint import pprint
from datetime import datetime
import topics
import json
import math
'''
class Corpus_all_topics(object):
    def __init__(self, path, preprocess_fn, stopwords):
        self.path = path
        self.stopwords = stopwords 
        self.preprocess_fn = preprocess_fn
        
    def __iter__(self):
        # iterates through all topics
        with open(self.path, 'r') as f:
            while True:
                doc = f.readline().strip()
                if doc == '':
                    break
                yield self.preprocess_fn(doc, self.stopwords)
'''
def compute_similarities(corpus_topic_ids, active_topic_ids, profile_words, 
                         preprocess_fn, stopwords, coeff, T, update, path):
    '''
    Computes the similarity scores between a topic profile and 
    each documents in the corpus
    Args:
    corpus_topic_ids: topic_ids to construct corpus from
    active_topic_ids: topic_ids to compute similarities for
    profile_words:    words representing a topic
    preprocess_fn:    function to preprocess original text 
    stopwords:        set of stopwords
    coeff:            contribution coefficient for in-document word frequency  
                      in computing word frequency in document
    T:                time attenuation factor
    update:           flag indicating whether this is an update operation
    path:             file path from which the stored dictionary is loaded,
                      used only when update=True
    Returns:
    Similarity matrix         
    '''
    if update:
        print('Updating similarity matrix...')
        with open(path, 'r') as f:
            similarities = json.load(f)
    else:
        print('Computing similarity matrix...')
        similarities = {}

    collection = topics.Topic_collection(corpus_topic_ids)
    collection.make_corpus(preprocess_fn, stopwords)
    collection.get_dictionary()
    #print(collection.dictionary.token2id)
    collection.get_distributions(coeff)
    for i, dst in enumerate(collection.distributions):
        if abs(sum(dst)-1) > 0.001:
            print(collection.topic_ids[i], sum(dst))

    for i, topic_id in enumerate(active_topic_ids):
        print(i)
        keywords = profile_words[topic_id]
        distribution = collection.get_distribution_given_profile(keywords)
        similarities[topic_id] = collection.get_similarity(distribution, T)

    print('dumping to JSON...')
    with open(path, 'w') as f:
        json.dump(similarities, f)

    return similarities

'''
_CORPUS = './sample_corpus.txt'
_STOPWORDS = './stopwords.txt'
stopwords = utils.load_stopwords(_STOPWORDS)

profile_words = {0:['雾', '霾'], 1:['股票']}

similarity_all = get_similarity_all(_CORPUS, utils.preprocess, stopwords, profile_words, 0.5)


print(similarity_all)

T = 365

tid_to_date = {141: '11/14/2017 9:47', 50: '12/4/2017 22:33'}

similarity_all = {141: {141: 0.01, 50: 1.035}, 50: {50: 0.008, 141: 0.748}}

pprint(similarity_all)

adjust_for_time(tid_to_date, similarity_all, T)

pprint(similarity_all)
'''




