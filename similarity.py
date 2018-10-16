from gensim import corpora
import utils
import copy
import numpy as np
from pprint import pprint
from datetime import datetime
import stream
import pickle
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
def compute_similarities(db, topic_ids, profile_words, preprocess_fn, stopwords,  
                         coeff, update, path):
    '''
    Computes the similarity scores between a topic profile and 
    each documents in the corpus
    Args:
    db:            database connection
    topic_ids:     topic_ids to compute similarities for
    profile_words: words representing a topic
    preprocess_fn: function to preprocess original text 
    stopwords:     set of stopwords
    coeff:         contribution coefficient for in-document word frequency  
                   in computing word frequency in document
    update:        flag indicating whether this is an update operation
    path:          file path from which the stored dictionary is loaded,
                   used only when update=True
    Returns:
    Similarity matrix         
    '''
    if update:
        print('Updating similarity matrix...')
        with open(path, 'rb') as f:
            similarities = pickle.load(f)
    else:
        print('Computing similarity matrix...')
        similarities = {}

    corpus = stream.Corpus_all_topics(db, preprocess_fn, stopwords)
    corpus.get_dictionary()
    print(corpus.dictionary.token2id)
    corpus.get_word_frequency()
    corpus.get_word_doc_prob(coeff)

    for topic_id in topic_ids:
        keywords = profile_words[topic_id]
        prob = corpus.get_prob_topic_profile(keywords)
        similarities[topic_id] = corpus.get_similarity(prob)

    with open(path, 'wb') as f:
        pickle.dump(similarities, f)

    return similarities

def adjust_for_time(tid_to_date, similarities, T, path):
    '''
    Adjust the similarity matrix for time difference
    Args:
    tid_to_date:    mapping from topic id to posting date
    similarity_all: Similarity matrix between between topics
    T:              time attenuation factor
    path:           file path from which the stored dictionary is loaded
    '''
    now = datetime.now()
    # construct an array of day differences between today and posting dates 
    # corresponding to topic corpus
    for topic_id, similarity in similarities.items():
        for topic_id_1 in similarity:
            if topic_id_1 not in tid_to_date:
                continue
            post_time = tid_to_date[topic_id_1]  
            similarity[topic_id_1] *= math.exp((now-post_time).days/T)

    with open(path, 'wb') as f:
        pickle.dump(similarities, f)

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




