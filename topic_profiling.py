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

def build_corpus(db_info, corpus_path_prefix=os.getcwd()):
    '''
    Builds raw corpus from the database and returns a mapping from 
    reply ID to corpus index
    Args: 
    db_info:              database information needed for connection
    corpus_path_prefix: path prefix for raw corpus files
    Returns:
    Dictionary mapping reply ID's to corpus indices for each topic 
    '''
    reply_id_to_corpus_index = {}
    try:
        db = pymysql.connect(*db_info)
        for i in range(10): 
            with db.cursor() as cursor_topic:
                sql_topic = 'SELECT TOPICID, BODY FROM topics_info_{}'.format(i)
                cursor_topic.execute(sql_topic)
                
                # outer loop over topics
                for (topic_id, content) in cursor_topic:
                    # ignore empty topics
                    if content is None or content.strip() == '':
                        continue

                    reply_id_to_corpus_index[topic_id] = {}
                    # for each topic create a raw corpus file
                    with open(os.path.join(corpus_path_prefix, 'topic_{}'.format(topic_id)), 'w') as f:                    
                        # the first line of corpus is always the topic content itself
                        f.write(' '.join(content.split()))
                        
                        # retrieve all replies under this topic ID
                        for j in range(10):
                            with db.cursor() as cursor_reply:
                                sql_reply = '''SELECT REPLYID, BODY FROM replies_info_{}
                                               WHERE TOPICID = {}'''.format(j, topic_id)
                                cursor_reply.execute(sql_reply)
                                # inner loop over replies to given topic
                                idx = 1 # replies start with index 1 in corpus
                                for (reply_id, content) in cursor_reply:
                                    if content is not None:
                                        reply_id_to_corpus_index[topic_id][reply_id] = idx
                                        f.write('\n'+' '.join(content.split()))
                                        idx += 1

    finally:
        db.close()

    return reply_id_to_corpus_index
                
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

def compute_scores(db_info, topic_ids, id_to_index, features, weights):
    '''
    Computes importance scores for replies under each topic
    Args:
    db_info:       database information needed for connection
    topic_ids:   list of all topic id's in the database
    id_to_index: dictionary mapping under each topic
    features:    attributes to include in importance evaluation
    weights:     weights associated with attributes in features
    Returns:
    importance scores for replies under each topic
    '''
    # normalize weights
    norm_weights = [wt/sum(weights) for wt in weights]
    # normalize features using min-max-scaler
    scaler = preprocessing.MinMaxScaler()

    scores = {}

    try:
        db = pymysql.connect(*db_info)
        for topic_id in topic_ids:
            scores[topic_id] = [None for _ in range(len(id_to_index[topic_id]))]
            for i in range(10):       
                with db.cursor() as cursor:
                    attrs = ', '.join(['REPLYID']+features)
                    sql = '''SELECT {} FROM replies_{}
                             WHERE TOPICID = {}'''.format(attrs, i, topic_id)
                    cursor.execute(sql)
                    replies = cursor.fetchall()
                    # normalize features using min-max scaler
                    for j in range(len(features)):
                        values = [reply[j+1] for reply in replies]
                        norm_features.append(scaler.fit_transform(values))
                    for j, (reply_id, ) in enumerate(replies):
                        corpus_index = id_to_index[topic_id][reply_id]
                        norm_feature_vector = [col[j] for col in norm_features]
                        scores[topic_id][corpus_index] = np.dot(norm_feature_vector, norm_weights)

    finally:
        db.close()

    return scores

def word_importance(topic_ids, stopwords_path, preprocess_fn, dictionary, 
                    model, normalize, reply_scores, path_prefix=os.getcwd(), alpha=0.65):
    '''
    Computes word importance in a weighted corpus
    Args:
    topic_ids:      list of all topic id's in the database
    path_prefix:    path prefix for raw corpus files
    stopwords_path: stopword file path
    preprocess_fn:  function to preprocess raw text
    dictionary:     gensim Dictionary object
    model:          language model to convert corpus to a desirable 
                    represention (e.g., tf-idf)
    normalize:      whether to normalize the representation obtained from model
    reply_scores:   list of reply scores under each topic
    alpha:          weight associated with topic content itself
    Returns:
    dict of word importance values
    '''
    word_weights = {}
    for topic_id in topic_ids:
        word_weights[topic_id] = collections.defaultdict(float)
        corpus_path = os.path.join(path_prefix, 'topic_{}'.format(topic_id))
        
        # create an iterator through raw corpus
        stream = Corpus_stream(corpus_path, stopwords_path, preprocess_fn, dictionary)
        language_model = model(stream, normalize=normalize)    

        # get the max score under each topic for normalization purposes
        max_score = max(reply_scores[topic_id])
        for i, text in enumerate(stream):
            converted = language_model[text]
            max_word_weight = max([x[1] for x in converted])
            coeff, score_norm = alpha, 1 if i == 0 else 1-alpha, reply_scores[topic_id][i]/max_score
            for word in converted:
                word_weight_norm = word[1]/max_word_weight
                word_weights[word[0]] += coeff*score_norm*word_weight_norm

    return word_weights

'''
corpus_path, stopwords_path = './corpus.txt', './stopwords.txt'
dictionary = build_dictionary(corpus_path, comment_scoring.preprocess, stopwords_path)

print(word_importance(corpus_path, stopwords_path, comment_scoring.preprocess, 
                      dictionary, models.TfidfModel, False))
'''
db_info = ('192.168.1.102','tgbweb','tgb123321','taoguba',3307)