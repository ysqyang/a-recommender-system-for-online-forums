from gensim import corpora, models, similarities
import jieba
import comment_scoring
import random
import collections
import pandas as pd
from sklearn import preprocessing
import numpy as np
import pymysql
import pickle

STOPWORDS = './stopwords.txt'
DB_INFO = ('192.168.1.102','tgbweb','tgb123321','taoguba',3307)
TOPIC_ID_TO_TABLE_NUM = './topic_id_to_table_num'
IMPORTANCE_FEATURES = [] 
WEIGHTS = []
SAVE_PATH = './word_importance'

class Corpus_stream(object):
    '''
    Corpus object for streaming and preprocessing 
    texts directly from the topics_info and replies_info tables
    '''
    def __init__(self, database, topic_id, topic_id_to_tbl_num, 
                 preprocess_fn):
        self.cursor = database.cursor()
        self.topic_id = topic_id
        self.topic_id_to_tbl_num = topic_id_to_tbl_num 
        self.preprocess_fn = preprocess_fn
        self.reply_id_to_corpus_index = {}
        
    def __iter__(self):
        # iteration starts with the topic content first
        table_num = self.topic_id_to_tbl_num[self.topic_id]
        sql = '''SELECT BODY FROM topics_info_{}
                 WHERE TOPICID = {}'''.format(table_num, self.topic_id)
        self.cursor.execute(sql)
        topic_content = ' '.join(self.cursor.fetchone().split())
        yield self.preprocess_fn(topic_content, STOPWORDS)
        
        # iterates through replies under this topic id
        for i in range(10):
            sql = '''SELECT REPLYID, BODY FROM replies_info_{}
                     WHERE TOPICID = {}'''.format(i, self.topic_id)
            self.cursor.execute(sql)
            idx = 1
            for (reply_id, content) in self.cursor:
                if content is not None:
                    self.reply_id_to_corpus_index[reply_id] = idx 
                    text = ' '.join(content.split())
                    idx += 1
                    yield self.preprocess_fn(text, STOPWORDS)

def establish_database_connection(db_info):
    '''
    Establishes a connection to the database specified by db_info
    Args:
    db_info: a tuple of (host, user, password, database, port)
    Returns:
    A database connection object
    '''
    return pymysql.connect(*db_info)


def build_dictionary(corpus_stream):
    '''
    Builds a dictionary from corpus_stream
    Args:
    corpus_stream: Corpus_stream object for a given topic 
    '''
    return corpora.Dictionary(corpus_stream)

def compute_scores(db, topic_id, features, weights, id_to_index):
    '''
    Computes importance scores for replies under each topic
    Args:
    db:          pymysql database connection 
    topic_id:    integer identifier for a topic
    features:    attributes to include in importance evaluation
    weights:     weights associated with attributes in features
    id_to_index: mapping from reply id to corpus index
    Returns:
    importance scores for replies
    '''
    # normalize weights
    norm_weights = [wt/sum(weights) for wt in weights]
    # normalize features using min-max-scaler
    scaler = preprocessing.MinMaxScaler()
    scores = {}

    for i in range(10):       
        with db.cursor() as cursor:
            attrs = ', '.join(['REPLYID']+features)
            sql = '''SELECT {} FROM replies_{}
                     WHERE TOPICID = {}'''.format(attrs, i, topic_id)
            cursor.execute(sql)
            replies = cursor.fetchall()
            # normalize features one by one using min-max scaler
            for j in range(len(features)):
                values = [reply[j+1] for reply in replies]
                norm_features.append(scaler.fit_transform(values))
            for j, (reply_id, ) in enumerate(replies):
                corpus_index = id_to_index[topic_id][reply_id]
                norm_feature_vector = [col[j] for col in norm_features]
                scores[corpus_index] = np.dot(norm_feature_vector, norm_weights)

    return scores

def word_importance(corpus_stream, dictionary, topic_id, model, 
                    normalize, scores, alpha=0.7):
    '''
    Computes word importance in a weighted corpus
    Args:
    corpus_stream: Corpus_stream object for a given topic 
    topic_id:      integer topic identifier
    model:         language model to convert corpus to a desirable 
                   represention (e.g., tf-idf)
    normalize:     whether to normalize the representation obtained from model
    scores:        list of reply scores under each topic
    alpha:         weight associated with topic content itself
    Returns:
    dict of word importance values
    '''
    word_weights = collections.defaultdict(float)

    corpus = [dictionary.doc2bow(doc) for doc in corpus_stream]
    language_model = model(corpus, normalize=normalize)    

    # get the max score under each topic for normalization purposes
    max_score = max(reply_scores)
    for i, doc in enumerate(corpus):
        converted = language_model[doc]
        max_word_weight = max([x[1] for x in converted])
        coeff, score_norm = 1-alpha, reply_scores[i]/max_score if i else alpha, 1 
        for word in converted:
            word_weight_norm = word[1]/max_word_weight
            word_weights[word[0]] += coeff*score_norm*word_weight_norm

    return word_weights

'''
corpus_path = './corpus.txt'
dictionary = build_dictionary(corpus_path, comment_scoring.preprocess, stopw)

print(word_importance(corpus_path, stopwords_path, comment_scoring.preprocess, 
                      dictionary, models.TfidfModel, False))
'''
def main():
    db = establish_database_connection(DB_INFO)
    tid_to_table = topic_id_to_table_number(db, TOPIC_ID_TO_TABLE_NUM)
    word_weights = {}

    # create a Corpus_stream object for each topic
    for topic_id in tid_to_table:
        stream = Corpus_stream(db, topic_id, tid_to_table, comment_scoring.preprocess)
        dictionary = build_dictionary(stream)
        scores = compute_scores(db, topic_id, IMPORTANCE_FEATURES, 
                                weights, stream.reply_id_to_corpus_index)        
        word_weights[topic_id] = word_weights(stream, dictionary, stream.topic_id, 
                                              model, normalize, scores)

    with open(SAVE_PATH, 'w') as f:
        pickle.dump(word_weights, f)

