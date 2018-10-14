from gensim import corpora, models
import collections
from sklearn import preprocessing
import stream
import numpy as np
import sys
import warnings
import pickle

def compute_profiles(db, topic_ids, tid_to_table, tid_to_reply_table, 
                     features, weights, preprocess_fn, stopwords, update,
                     path, alpha=0.7, smartirs='atn'):

    '''
    Computes topic profiles
    Args:
    db:                 database
    topic_ids:          list of topic id's to compute keyword profiles for       
    tid_to_table:       dictionary mapping topic id to topic table number
    tid_to_reply_table: dictionary mapping topic id to reply table number
    features:           attributes to include in importance evaluation
    weights:            weights associated with attributes in features
    preprocess_fn:      function to preprocess original text 
    stopwords:          set of stopwords
    update:             flag indicating whether this is an update operation
    path:               file path from which the stored dictionary is loaded,
                        used only when update=True
    alpha:              contribution coefficient for the topic content   
    smartirs:           tf-idf weighting variants

    Returns:
    Topics profiles in the form of word weight dictionaries
    '''
    if update:
        print('Updating word weights for active topics...')
        with open(path, 'rb') as f:
            word_weights = pickle.load(f)
    else:
        print('Computing word weights for all topics...')
        word_weights = {}
   
    for topic_id in topic_ids:
        if topic_id in tid_to_reply_table:
            reply_table_num = tid_to_reply_table[topic_id]
        else:
            reply_table_num = None
        # create a Corpus_under_topic object for each topic
        corpus = stream.Corpus_under_topic(db, topic_id, 
                                           tid_to_table[topic_id],
                                           reply_table_num, 
                                           preprocess_fn, stopwords)
        
        corpus.get_scores(db, topic_id, features, weights, 
                            corpus.reply_id_to_corpus_index, 
                            reply_table_num) 

        corpus.get_word_weight(alpha, smartirs)

        '''
        if i+1 == int(n_topic_ids*percentage):
            print('{}% finished'.format(percentage))
            percentage += .01
        '''
    with open(path, 'wb') as f:
        pickle.dump(word_weights, f)

    return word_weights

'''
def update_scores(db, active_topics, path, features, weights, rid_to_index, 
                  reply_table_num):
    
    with open(path, 'rb') as f:
        scores = pickle.load(f)

    for topic_id in active_topics:
        scores[topic_id] = get_scores(db, topic_id, features, weights, 
                                      rid_to_index, reply_table_num)

    with open(path, 'wb') as f:
        pickle.dump(scores, f)

    return scores
'''

def get_profile_words(topic_ids, word_weights, k, update, path):
    '''
    Get the top k most important words for a topic
    Args:
    word_weights: word weights for each topic
    k:            number of words to represent the topic
    update:       flag indicating whether this is an update operation
    path:         file path from which the stored dictionary is loaded,
                  used only when update=True
    Returns:
    Top k most important words for topics specified by topic_ids
    '''
    if update:
        with open(path, 'rb') as f:
            top_k = pickle.load(f)
    else:
        top_k = {}

    if k > len(word_weight):
        k = len(word_weight)

    for topic_id in topic_ids:
        weight = [(w, word_weight[w]) for w in word_weights[topic_id]]
        weight.sort(key=lambda x:x[1], reverse=True)
        top_k[topic_id] = weight[:k]

    with open(path, 'wb') as f:
        pickle.dump(top_k, f)

    return top_k 