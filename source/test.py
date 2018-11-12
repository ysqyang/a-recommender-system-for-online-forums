# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
import utils
import random
from gensim import corpora, models, similarities
import collections
from pprint import pprint
import time
from datetime import date, datetime, timedelta
import database
import pickle
import constants as const
import json
import database
import pika
import bisect
import collections
import logging
import os
import time
import requests

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.exchange_declare(exchange=const._EXCHANGE_NAME,
                         exchange_type='direct')

channel.queue_declare(queue='new_topics')
channel.queue_declare(queue='delete_topics')
#channel.queue_declare(queue='active_topics')

with open(const._TOPIC_FILE, 'r') as f:
    topics = json.load(f)

for tid, info in topics.items():
    t = datetime.strptime(info['POSTDATE'], const._DATETIME_FORMAT)
    topics[tid] = {'postDate': time.mktime(t.timetuple()),
                   'body': info['body']}

for tid in topics:
    rec = topics[tid]
    rec['topicID'] = tid
    msg = json.dumps(rec)
    channel.basic_publish(exchange=const._EXCHANGE_NAME,
                          routing_key='new',
                          body=msg)

#channel.basic_publish(exchange=const._EXCHANGE_NAME,
#                      routing_key='delete',
#                      body=msg2)

connection.close()
'''
with open(const._TOPIC_FILE, 'r') as f:
    topics = json.load(f)

tid = 1506377
print(topics[str(tid)]['body'])

query_dict = {'topicID': str(tid)}
r = requests.get('http://127.0.0.1:8000/serve/', params=query_dict)

print('*'*80)
print('您可能感兴趣的内容...')

recoms = r.json()

for tid in recoms:
    print('*'*80)
    print(topics[tid]['body'])

'''


