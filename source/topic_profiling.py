# -*- coding: utf-8 -*-

from gensim import corpora, models
import collections
import topics
import sys
import json

def compute_profiles(topic_ids, features, weights, preprocess_fn, stopwords,
                     update, path, alpha=0.7, smartirs='atn'):

    '''
    Computes topic profiles
    Args:
    topic_ids:          list of topic id's to compute keyword profiles for       
    features:           attributes to include in importance evaluation
    weights:            weights associated with attributes in features
    preprocess_fn:      function to preprocess original text 
    stopwords:          set of stopwords
    update:             flag indicating whether this is an update operation
    path:               file path for loading and saving
    alpha:              contribution coefficient for the topic content   
    smartirs:           tf-idf weighting variants

    Returns:
    Topics profiles in the form of word weight dictionaries
    '''
    if update:
        print('Updating word weights for active topics...')
        with open(path, 'r') as f:
            profiles = json.load(f)
    else:
        print('Computing word weights for all topics...')
        profiles = {}

    for topic_id in topic_ids:
        # create a Topic object for each topic
        topic_id = str(topic_id)
        topic = topics.Topic(topic_id)
        topic.make_corpus_with_scores(preprocess_fn, stopwords, features, weights)   
        if topic.valid:
            topic.get_dictionary()
            #print('scores for topic {}:'.format(topic.topic_id), topic.scores) 
            topic.get_word_weight(alpha, smartirs)
            profiles[topic_id] = topic.word_weight
        '''
        if i+1 == int(n_topic_ids*percentage):
            print('{}% finished'.format(percentage))
            percentage += .01
        '''
        #print('word_weights for topic id {}: '.format(topic_id), profiles[topic_id])
    with open(path, 'w') as f:
        json.dump(profiles, f)

    print('二次过滤后剩余{}条有效主贴'.format(len(profiles)))
    return profiles

'''
def update_scores(db, active_topics, path, features, weights, rid_to_index, 
                  reply_table_num):
    
    with open(path, 'r') as f:
        scores = json.load(f)

    for topic_id in active_topics:
        scores[topic_id] = get_scores(db, topic_id, features, weights, 
                                      rid_to_index, reply_table_num)

    with open(path, 'w') as f:
        json.dump(scores, f)

    return scores
'''

def get_profile_words(profiles, k, update, path):
    '''
    Get the top k most important words for a topic
    Args:
    profiles: word weights for each topic
    k:        number of words to represent the topic
    update:   flag indicating whether this is an update operation
    path:     file path for loading and saving
    Returns:
    Top k most important words for topics specified by topic_ids
    '''
    if update:
        with open(path, 'r') as f:
            top_k = json.load(f)
    else:
        top_k = {}

    for topic_id in profiles:
        profile = [(w, weight) for w, weight in profiles[topic_id].items()]
        profile.sort(key=lambda x:x[1], reverse=True)
        top_num = min(k, len(profile))
        top_k[topic_id] = [tup[0] for tup in profile[:top_num]]
    
    with open(path, 'w') as f:
        json.dump(top_k, f)

    return top_k 