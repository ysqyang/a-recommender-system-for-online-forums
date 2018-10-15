from gensim import corpora, models
import collections
import stream
import sys
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
    path:               file path for loading and saving
    alpha:              contribution coefficient for the topic content   
    smartirs:           tf-idf weighting variants

    Returns:
    Topics profiles in the form of word weight dictionaries
    '''
    if update:
        print('Updating word weights for active topics...')
        with open(path, 'rb') as f:
            profiles = pickle.load(f)
    else:
        print('Computing word weights for all topics...')
        profiles = {}
   
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
        
        corpus.get_dictionary()
        corpus.get_scores(features, weights, reply_table_num)
        print('scores for topic {}:'.format(corpus.topic_id), corpus.scores) 

        corpus.get_word_weight(alpha, smartirs)
        profiles[topic_id] = corpus.word_weight
        '''
        if i+1 == int(n_topic_ids*percentage):
            print('{}% finished'.format(percentage))
            percentage += .01
        '''
        #print('word_weights for topic id {}: '.format(topic_id), profiles[topic_id])
    with open(path, 'wb') as f:
        pickle.dump(profiles, f)

    return profiles

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

def get_profile_words(topic_ids, profiles, k, update, path):
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
        with open(path, 'rb') as f:
            top_k = pickle.load(f)
    else:
        top_k = {}

    if k > len(profiles):
        k = len(profiles)

    for topic_id in topic_ids:
        profile = [(w, weight) for w, weight in profiles[topic_id].items()]
        profile.sort(key=lambda x:x[1], reverse=True)
        top_k[topic_id] = [tup[0] for tup in profile[:k]]
        print('top k words for topic id {}: '.format(topic_id), top_k[topic_id])
    
    with open(path, 'wb') as f:
        pickle.dump(top_k, f)

    return top_k 