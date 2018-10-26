import numpy as np
from sklearn import preprocessing
import utils
import random
from gensim import corpora, models
import collections
from pprint import pprint
import time
from datetime import date, datetime
import database
import pickle
import constants as const
import json
import database
import pika
'''
class Stream(object):
    def __init__(self, topic_id, preprocess_fn, stopwords):
        self.topic_id = topic_id
        self.preprocess_fn = preprocess_fn
        self.stopwords = stopwords
        self.id_to_index = {}
        self.model = models.TfidfModel

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

    def create_dictionary(self):
        self.dictionary = corpora.Dictionary(self)

    def get_scores(self, results):
        self.scores = {}
        
        s, scaler = sum(weights), preprocessing.MinMaxScaler() 
        norm_weights = [wt/s for wt in weights]

        features_norm = scaler.fit_transform(np.array(results)[..., 1:])
        print('transformed: ', features_norm)
        for result, feature_vec in zip(results, features_norm):
            if result[0] in self.id_to_index:
                corpus_index = self.id_to_index[result[0]]
                self.scores[corpus_index] = np.dot(feature_vec, norm_weights)

    def get_word_weight(self, alpha=0.7, smartirs='atn'):
        self.word_weight = collections.defaultdict(float)
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in self]
        language_model = self.model(corpus_bow, smartirs=smartirs)

        # if there is no replies under this topic, use augmented term frequency
        # as word weight
        if len(corpus_bow) == 1 and len(corpus_bow[0]):
            max_freq = max(x[1] for x in corpus_bow[0])
            self.word_weight = {x[0]:(1+x[1]/max_freq)/2 for x in corpus_bow[0]}
            return

        # get the max score under each topic for normalization purposes
        max_score = max(self.scores.values())

        for i, doc in enumerate(corpus_bow):
            if len(doc) == 0:
                continue
            converted = language_model[doc]
            print(i, converted)
            if len(converted) == 0:
                continue
            max_word_weight = max([x[1] for x in converted])
            coeff = 1-alpha if i else alpha
            score_norm = self.scores[i]/max_score if i else 1 
            for word_id, weight in converted:
                word_weight_norm = weight/max_word_weight
                print(word_id, weight, coeff, score_norm, word_weight_norm)
                self.word_weight[word_id] += coeff*score_norm*word_weight_norm


topics = [22, 14]
weights = [1,4,2,5]
stopwords = utils.load_stopwords('./stopwords.txt')

results = {}
results[22] = [(95,)+tuple(random.randrange(10) for _ in range(4)),
              (173,)+tuple(random.randrange(10) for _ in range(4)),
              (315,)+tuple(random.randrange(10) for _ in range(4)), 
              (1004,)+tuple(random.randrange(10) for _ in range(4))
              ]

results[14] = [(11,)+tuple(random.randrange(10) for _ in range(4)),
              (263,)+tuple(random.randrange(10) for _ in range(4)),
              (477,)+tuple(random.randrange(10) for _ in range(4)), 
              (985,)+tuple(random.randrange(10) for _ in range(4))
              ]

for _id in topics:
    pprint(np.array(results[_id]))
    corpus = Stream(_id, utils.preprocess, stopwords)
    corpus.create_dictionary()
    pprint(corpus.id_to_index)
    pprint(corpus.dictionary.token2id)
    print(corpus.dictionary[7])
    corpus.get_scores(results[_id])
    corpus.get_word_weight()
    print('scores: ', corpus.scores)
    print('word weights: ', corpus.word_weight)


_STOPWORDS = 'stopwords.txt'

_TOPIC_ID_TO_TABLE_NUM = './topic_id_to_table_num'
_TOPIC_ID_TO_DATE = './topic_id_to_date'
_IMPORTANCE_FEATURES = ['USEFULNUM', 'GOLDUSEFULNUM', 'TOTALPCPOINT'] 
_WEIGHTS = [1, 1, 1]
_SAVE_PATH_WORD_IMPORTANCE = './word_importance'
_SAVE_PATH_SIMILARITY = './similarity'
_SAVE_PATH_SIMILARITY_ADJUSTED = './similarity_adjusted'

stopwords = utils.load_stopwords(_STOPWORDS)
print('stopwords loaded')
db = utils.connect_to_database(_DB_INFO)
print('connection to database established')
tid_to_table = utils.load_topic_id_to_table_num(db, _TOPIC_ID_TO_TABLE_NUM)
print('topic-id-to-table-number mapping loaded')
tid_to_date = utils.load_topic_id_to_date(db, _TOPIC_ID_TO_DATE)
print('topic-id-to-post-date mapping loaded')

print(len(tid_to_table),len(tid_to_date))
#print(tid_to_table.keys())

tid_to_table = utils.load_mapping(const._TOPIC_ID_TO_TABLE_NUM)
db = database.Database(*const._DB_INFO)
active_topics = utils.get_new_topics(db, tid_to_table)
tid_to_table = utils.update_tid_to_table_num_mapping(const._TOPIC_ID_TO_TABLE_NUM, db, active_topics)
tid_to_reply_table = utils.update_tid_to_reply_table_num_mapping(const._TOPIC_ID_TO_REPLY_TABLE_NUM, db, active_topics)
_DB_INFO = ('192.168.1.102','tgbweb','tgb123321','taoguba', 3307, 'utf8mb4')

db = database.Database(*const._DB_INFO)
topic_ids = utils.load_topics(db, const._TOPIC_FEATURES, const._DAYS, const._TOPIC_FILE)
utils.load_replies(db, topic_ids, const._FEATURES, const._REPLY_FILE)

with open(const._REPLY_FILE, 'r') as f:
    replies = json.load(f)

with open(const._TOPIC_FILE, 'r') as f:
    topics = json.load(f)

n_replies = 0
for topic_id, r in replies.items():
    n_replies += len(r)

print(n_replies)

n_replies = 0
for topic_id, r in topics.items():
    n_replies += r['TOTALREPLYNUM']

print(n_replies)


connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.exchange_declare(exchange='x',
                         exchange_type='direct')

channel.queue_declare(queue='new_topics')
channel.queue_declare(queue='update_topics')
channel.queue_declare(queue='active_topics')

d1 = {'topicid': '1600003', 'TOTALVIEWNUM': 9, 'TOTALREPLYNUM': 0, 
     'POSTDATE': '2018-10-26', 'USEFULNUM': 0, 'GOLDUSEFULNUM': 0, 
     'TOTALPCPOINT': 0, 'TOPICPCPOINT': 0, 'body': 'dsfghfd'}

d2 = {'topicid': '1506315', 'TOTALVIEWNUM': 3, 'USEFULNUM': 1}

d3 = {}

msg1, msg2 = json.dumps(d1), json.dumps(d2)

channel.basic_publish(exchange='x',
                      routing_key='new',
                      body=msg1)

channel.basic_publish(exchange='x',
                      routing_key='update',
                      body=msg2)

connection.close()
'''

dt1 = datetime.strptime('2018-10-25 00:00:00', "%Y-%m-%d %H:%M:%S")
dt2 = datetime.strptime('2018-10-25 23:59:59', "%Y-%m-%d %H:%M:%S")
print(dt1.date() > dt2.date())