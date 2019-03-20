import os
import sys
from datetime import datetime, timedelta
import time
from collections import defaultdict
import argparse
import threading
import logging
import json
import math
#import topic_profiling as tp
#import similarity as sim
from gensim.models import TfidfModel
import pika
from classes TextPreprocessor, CorpusSimilarity, CorpusTfidf, Recom
import utils
root_dir = os.path.dirname(sys.path[0])
config_path = os.path.abspath(os.path.join(root_dir, 'config'))
sys.path.insert(1, config_path)
import constants as const
import log_config as lc
import mq_config as mc


class Save(threading.Thread):
    def __init__(self, topics, recoms, interval, lock, topic_dir, 
                 recom_dir, mod_num, logger=None):
        threading.Thread.__init__(self)
        self.topics = topics
        self.recoms = recoms  # recommendations for special topics
        self.interval = interval
        self.lock = lock
        self.topic_dir = topic_dir
        self.recom_dir = recom_dir
        self.mod_num = mod_num
        self.logger = logger

    def run(self):
        while True:
            time.sleep(self.interval)
            with self.lock: 
                self._save_topics()
                self._save_recoms()
                 
    def _save_topics(self):
        self.topics.save(self.topic_dir, self.mod_num)

    def _save_recoms(self):
        if not os.path.exists(self.recom_dir):
            os.makedirs(self.recom_dir)
        for stid in self.recoms:
            with open(os.path.join(self.recom_dir, stid), 'w') as f:
                json.dump(self.recoms[stid], f)


class Delete(threading.Thread):
    def __init__(self, topics, interval, keep_days, lock, logger=None):
        threading.Thread.__init__(self)
        self.topics = topics
        self.interval = interval
        self.keep_days = keep_days
        self.lock = lock
        self.logger = logger

    def run(self):
        while True:
            time.sleep(self.interval)
            if self.topics.size == 0:
                return
            with self.lock:
                latest = datetime.fromtimestamp(self.topics.data[self.topics.latest]['date'])
                t = latest - timedelta(days=self.keep_days)
                self.logger.info('Removing all topics older than {}'.format(t))
                self.topics.remove_before(t)

'''
class Delete(threading.Thread):
    def __init__(self, topics, save_dir):
        threading.Thread.__init__(self)
        self.topics = topics
        self.interval = interval
        self.save_dir = save_dir

    def run(self):
        while True:
            time.sleep(self.interval)
            with self.lock:
'''


def main(args):  
    while True:
        try:
            logger = utils.get_logger_with_config(name=lc.RUN_LOG_NAME,
                                                  logger_level=lc.LOGGER_LEVEL,
                                                  handler_levels=lc.LEVELS,
                                                  log_dir=lc.LOG_DIR,
                                                  mode=lc.MODE,
                                                  log_format=lc.LOG_FORMAT)
            break
        except Exception as e:
            logging.exception(e)

    # load stopwords
    stopwords = utils.load_stopwords(const.STOPWORD_FILE)

    preprocessor = TextPreprocessor(singles=const.SINGLES,
                                    puncs=const.PUNCS,
                                    punc_frac_low=const.PUNC_FRAC_LOW,
                                    punc_frac_high=const.PUNC_FRAC_HIGH,
                                    valid_count=const.VALID_COUNT,
                                    valid_ratio=const.VALID_RATIO,
                                    stopwords=stopwords)

    topics = CorpusSimilarity(name='topics',
                             time_decay_scale=const.TIME_DECAY_SCALE,
                             duplicate_thresh=const.DUPLICATE_THRESH,
                             irrelevant_thresh=const.IRRELEVANT_THRESH,
                             max_size=const.MAX_SIZE,
                             logger=utils.get_logger(lc.RUN_LOG_NAME+'.topics')
                             )

    specials = CorpusTfidf(name='specials',
                           logger=utils.get_logger(lc.RUN_LOG_NAME+'.specials')
                           )

    recoms = Recom(corpus_kw=specials,
                   corpus_target=topics,
                   max_len=const.MAX_LEN,
                   time_decay=const.)
    keyword_weight = defaultdict(list)

    # load previously saved corpus and similarity data if possible
    if args.l:
        try:
            topics.load(const.TOPIC_DIR)
        except FileNotFoundError:
            logger.exception('Topic data files not found. New files will be created')
        try:
            specials.load(const.SPECIAL_DIR)
        except FileNotFoundError:
            logger.exception('Special topic data files not found. New files will be created')

        files = os.listdir(const.RECOM_DIR)
        for file in files:
            if not file.isnumeric():
                continue
            path = os.path.join(const.RECOM_DIR, file)
            try:
                with open(path, 'r') as f: 
                    recoms[file] = json.load(f)
            except FileNotFoundError:
                logger.exception('File not found for special topic %s', file)
            except json.JSONDecodeError:
                logger.error('Failed to load topic %s', file)
    
    #print(topics.dictionary)

    # establish rabbitmq connection and declare queues
    if args.c:
        credentials = pika.PlainCredentials(username=mc.USERNAME,
                                            password=mc.PASSWORD)
        params = pika.ConnectionParameters(host=mc.HOST,
                                           credentials=credentials)
    else:
        params = pika.ConnectionParameters(host='localhost')

    lock = threading.Lock()
    save_topics = Save(topics=topics,
                       recoms=recoms,
                       interval=const.SAVE_EVERY,
                       lock=lock,
                       topic_dir=const.TOPIC_DIR,
                       recom_dir=const.RECOM_DIR,
                       mod_num=const.NUM_RESULT_DIRS)
    
    save_topics.start()

    delete_topics = Delete(topics=topics,
                           interval=const.DELETE_EVERY,
                           keep_days=const.KEEP_DAYS,
                           lock=lock,
                           logger=utils.get_logger(lc.RUN_LOG_NAME+'.topics'))

    delete_topics.start()
    
    while True:       
        try:
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.basic_qos(prefetch_count=1)
            channel.exchange_declare(exchange=const.EXCHANGE_NAME, 
                                     exchange_type='direct')
          
            channel.queue_declare(queue='new_topics')
            channel.queue_declare(queue='old_topics')
            channel.queue_declare(queue='special_topics')
            channel.queue_declare(queue='delete_topics')
            #channel.queue_declare(queue='update_topics')   
            channel.queue_bind(exchange=const.EXCHANGE_NAME, 
                               queue='new_topics', routing_key='new')
            channel.queue_bind(exchange=const.EXCHANGE_NAME,
                               queue='old_topics', routing_key='old')
            channel.queue_bind(exchange=const.EXCHANGE_NAME,
                               queue='special_topics', routing_key='special')
            channel.queue_bind(exchange=const.EXCHANGE_NAME, 
                               queue='delete_topics', routing_key='delete')
            #channel.queue_bind(exchange=const.EXCHANGE_NAME, queue='update_topics', routing_key='update')
            
            def decode_to_dict(msg):
                while type(msg) != dict:
                    msg = json.loads(msg)
                return msg

            def get_topic_data(topic):
                topic = decode_to_dict(topic)
                topic_id = str(topic['topicID'])
                content = preprocessor.preprocess(topic['body'])
                date = topic['postDate'] / const.TIMESTAMP_FACTOR

                return topic_id, content, date

            def on_new_topic(ch, method, properties, body):
                topic_id, content, date = get_topic_data(body)
                logger.info('Received new topic %s', topic_id)

                with lock:
                    if topics.add(topic_id, content, date):
                        for stid in specials.data:
                            match_val = get_relevance(stid, topic_id)
                            recom_insert(recoms[stid], topic_id, match_val)

                channel.basic_ack(delivery_tag=method.delivery_tag)      

            def on_old_topic(ch, method, properties, body):
                topic_id, content, date = get_topic_data(body)
                logger.info('Received old topic %s', topic_id)
                channel.basic_ack(delivery_tag=method.delivery_tag)

                with lock:
                    sim_list = topics.find_most_similar(old_topic)

                sim_list = [tid for tid, val in sim_list][:const.TOP_NUM]
                
                channel.basic_publish(exchange=const.EXCHANGE_NAME,
                                      routing_key='old',
                                      body=json.dumps(sim_list))

            def on_special_topic(ch, method, properties, body):
                topic_id, content, date = get_topic_data(body)
                logger.info('Received special topic %s', topic_id)

                with lock:
                    if specials.add(topic_id, content, date):
                        specials.generate_keywords(const.KEYWORD_NUM)

                        for stid in specials.data:
                            recom_list = [[tid, get_relevance(stid, tid)] for tid in topics.data]
                            recom_list.sort(key=lambda x:x[1], reverse=True)
                            recoms[stid] = recom_list
                
                channel.basic_ack(delivery_tag=method.delivery_tag) 

            def on_delete(ch, method, properties, body):
                topic_id, _, _ = get_topic_data(body)
                logger.info('Deleting topic %s', topic_id)
                
                with lock:
                    topics.delete(topic_id)
                channel.basic_ack(delivery_tag=method.delivery_tag)
            
            '''
            def on_update_topic(ch, method, properties, body):
                update_topic = json.loads(body)
                topic_id = update_topic['topicID']
                for attr in update_topic:
                    if attr != 'topicID':
                        topic_dict[topic_id][attr] = update_topic[attr]
                utils.save_topics(topic_dict, const.TOPIC_FILE)
            '''  

            channel.basic_consume('new_topics', on_new_topic)
            channel.basic_consume('special_topics', on_special_topic)
            channel.basic_consume('delete_topics', on_delete)
            channel.basic_consume('old_topics', on_old_topic)
            '''
            channel.basic_consume(on_update_topic, queue='update_topics')                                  
            '''    
            logger.info(' [*] Waiting for messages. To exit press CTRL+C')
            channel.start_consuming()
        
        except Exception as e:
            logger.exception(e)
            logger.info('Retrying in %d seconds', const.SLEEP_TIME)
            time.sleep(const.SLEEP_TIME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', action='store_true', help='load previously saved corpus and similarity data')
    parser.add_argument('-c', action='store_true', help='load message queue connection configurations from file')   
    args = parser.parse_args()
    main(args)