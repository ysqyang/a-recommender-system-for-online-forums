import os
import sys
from datetime import datetime, timedelta
import time
from collections import defaultdict
import argparse
import threading
import logging
import json
import yaml
import pika
from classes import TextPreprocessor, CorpusSimilarity, CorpusTfidf
import utils
root_dir = os.path.dirname(sys.path[0])
config_path = os.path.abspath(os.path.join(root_dir, 'config'))
sys.path.insert(1, config_path)


class Save(threading.Thread):
    def __init__(self, topics, specials, interval, lock, topic_path,
                 specials_path, mod_num, logger=None):
        threading.Thread.__init__(self)
        self.topics = topics
        self.specials = specials
        self.interval = interval
        self.lock = lock
        self.topic_path = topic_path
        self.specials_path = specials_path
        self.mod_num = mod_num
        self.logger = logger

    def run(self):
        while True:
            time.sleep(self.interval)
            with self.lock: 
                if not os.path.exists(self.topic_path):
                    os.makedirs(self.topic_path)
                self.topics.save(self.topic_path, self.mod_num)

                if not os.path.exists(self.specials_path):
                    os.makedirs(self.specials_path)
                self.specials.save(self.specials_path, self.mod_num)


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


def main(args):  
    # read configurations
    while True:
        try:
            with open('../config/config.yml', 'rb') as f:
                config = yaml.load(f)
                break
        except Exception as e:
            logging.exception(e)

    path_cfg = config['paths']
    main_cfg = config['main']
    log_cfg = config['logging']
    pre_cfg = config['preprocessing']
    recom_cfg = config['recommendation']
    mq_cfg = config['message_queue']
    misc_cfg = config['micellaneous']
    special_cfg = config['special_topics']
    logger = utils.get_logger_with_config(name=log_cfg['run_log_name'],
                                          logger_level=log_cfg['log_level'],
                                          handler_levels=log_cfg['handler_levels'],
                                          log_dir=log_cfg['dir'],
                                          mode=log_cfg['mode'],
                                          log_format=log_cfg['format'])


    # load stopwords
    stopwords = utils.load_stopwords(path_cfg['stopwords'])

    preprocessor = TextPreprocessor(singles=pre_cfg['singles'],
                                    puncs=pre_cfg['punctuations'],
                                    punc_frac_low=pre_cfg['min_punc_frac'],
                                    punc_frac_high=pre_cfg['max_punc_frac'],
                                    valid_count=pre_cfg['min_count'],
                                    valid_ratio=pre_cfg['min_ratio'],
                                    stopwords=stopwords)

    topics = CorpusSimilarity(name='TOPICS',
                              time_decay=recom_cfg['time_decay_base'],
                              duplicate_thresh=recom_cfg['duplicate_thresh'],
                              irrelevant_thresh=recom_cfg['irrelevant_thresh'],
                              max_recoms=recom_cfg['max_recoms_stored'],
                              logger=utils.get_logger(log_cfg['run_log_name']+'.topics')
                              )

    specials = CorpusTfidf(name='SPECIAL TOPICS',
                           target_corpus=topics,
                           tfidf_scheme=special_cfg['smartirs_scheme'],
                           num_keywords=special_cfg['num_keywords'],
                           time_decay=recom_cfg['time_decay_base'],
                           max_recoms=recom_cfg['max_recoms_stored_special'],
                           logger=utils.get_logger(log_cfg['run_log_name']+'.specials')
                           )

    # load previously saved corpus and similarity data if possible
    if args.l:
        try:
            topics.load(path_cfg['topics'])
        except FileNotFoundError:
            logger.exception('Topic data files not found. New files will be created')
        try:
            specials.load(path_cfg['special_topics'])
        except FileNotFoundError:
            logger.exception('Special topic data files not found. New files will be created')

    # establish rabbitmq connection and declare queues
    if args.c:
        credentials = pika.PlainCredentials(username=mq_cfg['username'],
                                            password=mq_cfg['password'])
        params = pika.ConnectionParameters(host=mq_cfg['host'],
                                           credentials=credentials)
    else:
        params = pika.ConnectionParameters(host='localhost')

    lock = threading.Lock()
    save_topics = Save(topics=topics,
                       specials=specials,
                       interval=main_cfg['save_every'],
                       lock=lock,
                       topic_path=path_cfg['topic_save_dir'],
                       specials_path=path_cfg['special_save_dir'],
                       mod_num=misc_cfg['num_result_dirs'])
    
    save_topics.start()

    delete_topics = Delete(topics=topics,
                           interval=main_cfg['delete_every'],
                           keep_days=main_cfg['keep_days'],
                           lock=lock,
                           logger=utils.get_logger(log_cfg['run_log_name']+'.topics'))

    delete_topics.start()
    
    while True:       
        try:
            exchange = mq_cfg['exchange_name']
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.basic_qos(prefetch_count=1)
            channel.exchange_declare(exchange=mq_cfg['exchange_name'], 
                                     exchange_type='direct')
          
            channel.queue_declare(queue='new_topics')
            channel.queue_declare(queue='old_topics')
            channel.queue_declare(queue='special_topics')
            channel.queue_declare(queue='delete_topics')

            channel.queue_bind(exchange=exchange, 
                               queue='new_topics', routing_key='new')
            channel.queue_bind(exchange=exchange,
                               queue='old_topics', routing_key='old')
            channel.queue_bind(exchange=exchange,
                               queue='special_topics', routing_key='special')
            channel.queue_bind(exchange=exchange, 
                               queue='delete_topics', routing_key='delete')
            
            def decode_to_dict(msg):
                while type(msg) != dict:
                    msg = json.loads(msg)
                return msg

            def get_topic_data(topic):
                topic = decode_to_dict(topic)
                topic_id = str(topic['topicID'])
                content = preprocessor.preprocess(topic['body']) if 'body' in topic else []
                date = topic['postDate']//misc_cfg['timestamp_factor'] if 'postDate' in topic else -1

                return topic_id, content, date

            def on_new_topic(ch, method, properties, body):
                topic_id, content, date = get_topic_data(body)

                with lock:
                    topics.add(topic_id, content, date)
                    specials.update_on_new_topic(topic_id, content, date)

                channel.basic_ack(delivery_tag=method.delivery_tag)      

            def on_old_topic(ch, method, properties, body):
                topic_id, content, date = get_topic_data(body)
                logger.info('Received old topic %s', topic_id)
                channel.basic_ack(delivery_tag=method.delivery_tag)

                with lock:
                    sim_list = topics.find_most_similar(content)

                sim_list = [tid for tid, val in sim_list][recom_cfg['max_recoms_stored']]
                
                channel.basic_publish(exchange=exchange,
                                      routing_key='old',
                                      body=json.dumps(sim_list))

            def on_special_topic(ch, method, properties, body):
                topic_id, content, date = get_topic_data(body)

                with lock:
                    specials.add(topic_id, content, date)
                
                channel.basic_ack(delivery_tag=method.delivery_tag) 

            def on_delete(ch, method, properties, body):
                topic_id, _, _ = get_topic_data(body)
                
                with lock:
                    specials.update_on_delete_topic(topic_id)
                    topics.delete(topic_id)

                channel.basic_ack(delivery_tag=method.delivery_tag)

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
            logger.info('Retrying in %d seconds', main_cfg['retry_every'])
            time.sleep(main_cfg['retry_every'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', action='store_true', help='load previously saved corpus and similarity data')
    parser.add_argument('-c', action='store_true', help='load message queue connection configurations from file')   
    args = parser.parse_args()
    main(args)