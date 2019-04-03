import numpy as np
from sklearn import preprocessing
import utils
import random
from gensim import corpora, models, similarities, matutils
import collections
from pprint import pprint
import time
from datetime import date, datetime, timedelta
import pickle
import json
import jieba
import pika
import bisect
import collections
import logging
import os, sys
import time
import re
import argparse
import requests
import yaml
root_dir = os.path.dirname(sys.path[0])
config_path = os.path.abspath(os.path.join(root_dir, 'config'))
sys.path.insert(1, config_path)


def main(args):
    with open('../config/config.yml', 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        path_cfg = config['paths']
        main_cfg = config['main']
        log_cfg = config['logging']
        pre_cfg = config['preprocessing']
        recom_cfg = config['recommendation']
        mq_cfg = config['message_queue']
        misc_cfg = config['micellaneous']
        special_cfg = config['special_topics']

    if args.p:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()
        channel.exchange_declare(exchange=mq_cfg['exchange_name'], exchange_type='direct')
        channel.queue_declare(queue='new_topics')
        channel.queue_declare(queue='special_topics')
        channel.queue_declare(queue='delete_topics')
        channel.queue_declare(queue='old_topics')
        #channel.queue_declare(queue='active_topics')
        with open(path_cfg['topics'], 'r') as f:
            topics = json.load(f)
        for tid, info in topics.items():
            t = datetime.strptime(info['POSTDATE'], misc_cfg['datetime_format'])
            topics[tid] = {'postDate': time.mktime(t.timetuple())*1000,
                           'body': info['body']}
        tids = sorted(list(topics.keys()))
        for tid in tids:
            rec = topics[tid]
            rec['topicID'] = tid
            msg = json.dumps(rec)
            if tid in {'1506377', '1506414'}:
                channel.basic_publish(exchange=mq_cfg['exchange_name'],
                                      routing_key='special',
                                      body=msg)            
            else:
                channel.basic_publish(exchange=mq_cfg['exchange_name'],
                                      routing_key='new',
                                      body=msg)

        msg = json.dumps({'topicID': '1506556'})
        channel.basic_publish(exchange=mq_cfg['exchange_name'],
                              routing_key='delete',
                              body=msg)

        connection.close()
    else:       
        with open(path_cfg['topics'], 'r') as f:
            topics = json.load(f)
        tid = 1506404
        print(topics[str(tid)]['body'])
        query_dict = {'topicID': str(tid)}
        if args.s:
            r = requests.get('http://127.0.0.1:8000/serve/', params=query_dict)
        else:
            r = requests.get('http://127.0.0.1:8000/serve_special/', params=query_dict)
        print('*'*80)
        print('您可能感兴趣的内容...')
        response = r.json()
        recoms = response['dto']['list']
        print(recoms)
        for tid, match_val in recoms:
            print('*'*80)
            print(topics[tid]['body'], match_val)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store_true', help='producing messages')
    parser.add_argument('-s', action='store_true', help='serve recommendations')   
    args = parser.parse_args()
    main(args)