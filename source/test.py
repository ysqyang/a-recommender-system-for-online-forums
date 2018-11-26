import numpy as np
from sklearn import preprocessing
import utils
import random
from gensim import corpora, models, similarities
import collections
from pprint import pprint
import time
from datetime import date, datetime, timedelta
import pickle
import json
import pika
import bisect
from collections import defaultdict
import logging
import os, sys
import time
import requests
root_dir = os.path.dirname(sys.path[0])
config_path = os.path.abspath(os.path.join(root_dir, 'config'))
sys.path.insert(1, config_path)
import constants as const


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
    topics[tid] = {'postDate': time.mktime(t.timetuple())*1000,
                   'body': info['body']}

tids = sorted(list(topics.keys()), reverse=True)
#print(tids)

for tid in tids:
    rec = topics[tid]
    rec['topicID'] = tid
    msg = json.dumps(rec)
    #print(msg)
    channel.basic_publish(exchange=const._EXCHANGE_NAME,
                            routing_key='new',
                            body=msg)

connection.close()
'''
with open(const._TOPIC_FILE, 'r') as f:
    topics = json.load(f)

tid = 1506380
print(topics[str(tid)]['body'])

query_dict = {'topicID': str(tid)}
r = requests.get('http://127.0.0.1:8000/serve/', params=query_dict)

print('*'*80)
print('您可能感兴趣的内容...')

response = r.json()

recoms = response['dto']['list']

for tid in recoms:
    print('*'*80)
    print(topics[tid]['body'])


class Foo(object):
    def __init__(self, vals):
        self.vals = vals

    def f(self):
        for i in range(len(self.vals)):
            self.vals[i] *= 2

class Foo1(Foo):
    def __init__(self, vals, x):
        super().__init__(vals)
        self.x = x

    def f(self):
        super().f()
        for i in range(len(self.vals)):
            self.vals[i] += self.x

o = Foo1([3,5, 7, 9], 1)
o.f()

print(o.vals)
'''



