from gensim import corpora, models, similarities
import jieba
import comment_scoring
import random
import collections
import pandas as pd
from sklearn import preprocessing
import numpy as np
import pymysql

class Corpus_stream(object):
    '''
    Corpus object for streaming preprocessed texts
    '''
    def __init__(self, corpus_path, stopwords_path, preprocess_fn, dictionary):
        self.corpus_path = corpus_path
        self.stopwords_path = stopwords_path
        self.preprocess_fn = preprocess_fn
        self.dictionary = dictionary

    def __iter__(self):
        with open(self.corpus_path, 'r') as f:
            while True:
                raw_text = f.readline().strip()
                if raw_text == '':
                    break
                yield self.dictionary.doc2bow(
                    self.preprocess_fn(raw_text, self.stopwords_path))

def build_corpus(db_info, out_file_path_prefix):
    '''
    Builds raw corpus from the database and returns a mapping from 
    reply ID to corpus index
    Args: 
    db_info:              database information needed for connection
    out_file_path_prefix: path prefix for raw corpus files
    Returns:
    
    '''
    reply_id_to_corpus_index = {}
    try:
        db = pymysql.connect(*db_info)
        with db.cursor() as cursor_topic:
            sql_topic = 'SELECT TOPICID, BODY FROM topics_info_{}'.format()
            cursor_topic.execute(sql_topic)
            # outer loop over topics
            while True:
                topic = cursor_topic.fetchone()
                with open(os.join(out_file_path_prefix, 'topic_{}'.format()), 'w') as f:
                    topic_id = topic[0]
                    f.write(topic[1])
                    # retrieve all replies under this topic ID
                    with db.cursor() as cursor_reply:
                        sql_reply = '''SELECT REPLYID, BODY FROM reply_info
                                       WHERE TOPICID = {}'''.format(topic_id)
                        cursor_reply.execute(sql_reply)
                        # inner loop over replies to given topic
                        while True:
                            reply = cursor_reply.fetchone()
                            f.write(reply[1].replace('\n', '')+'\n')

    finally:
        connection.close()
                

def build_dictionary(corpus_path, preprocess_fn, stopwords_path):
    return corpora.Dictionary(preprocess_fn(line.rstrip(), stopwords_path) 
                              for line in open(corpus_path, 'r')) 

def compute_scores(df, features):
    '''
    Computes scores for each text
    Args:
    df:       Pandas dataframe object containing original database
    features: list of attributes to include in computing scores
    weights:  list of weights for the attributes in features
    '''
    # normalize weights
    norm_weights = [wt/sum(weights) for wt in weights]

    for feature in features:
        print(feature, max(df[feature]), min(df[feature]))

    # normalize features using min-max-scaler
    scaler = preprocessing.MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

def word_importance(corpus_path, stopwords_path, preprocess_fn, 
                    dictionary, model, normalize):
    '''
    Computes word importance in a weighted corpus
    Args:
    corpus_path:    raw text file path
    stopwords_path: stopword file path
    preprocess_fn:  function to preprocess raw text
    dictionary:     gensim Dictionary object
    model:          language model to convert corpus to a desirable 
                    represention (e.g., tf-idf)
    normalize:      whether to normalize the representation obtained from model
    Returns:
    dict of word importance values
    '''
    for id_ in dictionary:
        print(id_, dictionary[id_])
    stream = Corpus_stream(corpus_path, stopwords_path, preprocess_fn, dictionary)
    language_model = model(stream, normalize=normalize)

    word_weights = collections.defaultdict(float)
    for text in stream:
        converted = language_model[text]
        max_word_weight = max([x[1] for x in converted])
        for word in converted:
            word_weight_norm = word[1]/max_word_weight
            word_weights[word[0]] += word_weight_norm

    return word_weights

'''
corpus_path, stopwords_path = './corpus.txt', './stopwords.txt'
dictionary = build_dictionary(corpus_path, comment_scoring.preprocess, stopwords_path)
'''

