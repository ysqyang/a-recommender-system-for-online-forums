import utils
#import topic_profiling as tp
#import similarity as sim
#from gensim import models
import json
import pika
import os, sys
import classes
from datetime import datetime
import time
import argparse
import threading
import logging
root_dir = os.path.dirname(sys.path[0])
config_path = os.path.abspath(os.path.join(root_dir, 'config'))
sys.path.insert(1, config_path)
import constants as const
import log_config as lc
import mq_config as mc

class Save(threading.Thread):
    def __init__(self, collection, interval, lock, save_dir, mod_num, logger=None):
        threading.Thread.__init__(self)
        self.collection = collection
        self.interval = interval
        self.lock = lock
        self.save_dir = save_dir
        self.mod_num = mod_num

    def run(self):
        while True:
            time.sleep(self.interval)
            with self.lock:
                self.collection.save(self.save_dir, self.mod_num)
'''
class Delete(threading.Thread):
    def __init__(self, collection, save_dir):
        threading.Thread.__init__(self)
        self.collection = collection
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
            logger = utils.get_logger_with_config(name           = lc._RUN_LOG_NAME, 
                                                  logger_level   = lc._LOGGER_LEVEL, 
                                                  handler_levels = lc._LEVELS,
                                                  log_dir        = lc._LOG_DIR, 
                                                  mode           = lc._MODE, 
                                                  log_format     = lc._LOG_FORMAT)
            break
        except Exception as e:
            logging.exception(e)

    # load stopwords
    stopwords = utils.load_stopwords(const._STOPWORD_FILE)

    collection = classes.Corpus_with_similarity_data( 
                         puncs             = const._PUNCS,
                         singles           = const._SINGLES, 
                         stopwords         = stopwords, 
                         punc_frac_low     = const._PUNC_FRAC_LOW,
                         punc_frac_high    = const._PUNC_FRAC_HIGH, 
                         valid_count       = const._VALID_COUNT, 
                         valid_ratio       = const._VALID_RATIO,
                         trigger_days      = const._TRIGGER_DAYS,
                         keep_days         = const._KEEP_DAYS, 
                         T                 = const._T,
                         irrelevant_thresh = const._IRRELEVANT_THRESH, 
                         logger            = utils.get_logger(lc._RUN_LOG_NAME+'.topics')
                         )

    collection.get_dictionary()

    # load previously saved corpus and similarity data if possible
    if args.l:
        try:
            collection.load(const._CORPUS_FOLDER)
        except FileNotFoundError as e:
            logger.exception('Data files not found. New files will be created')
    
    # establish rabbitmq connection and declare queues
    if args.c:
        while True:
            try:
                credentials = pika.PlainCredentials(username = mc._USERNAME,
                                                    password = mc._PASSWORD)
                
                params = pika.ConnectionParameters(host        = mc._HOST, 
                                                   credentials = credentials)
                break
            except Exception as e:
                logger.exception(e)
    else:
        params = pika.ConnectionParameters(host='localhost')

    '''
    def on_subject_update(ch, method, properties, body):
        while type(body) != dict:
            body = json.loads(body)

        subject_dict = body
        logger.info('Updating recommendations for subject %s', subject_dict['subID'])
        keyword_weight
        recoms = collection.get_topics_by_keywords(keyword_weight)
    
    def on_update_topic(ch, method, properties, body):
        update_topic = json.loads(body)
        topic_id = update_topic['topicID']
        for attr in update_topic:
            if attr != 'topicID':
                topic_dict[topic_id][attr] = update_topic[attr]
        utils.save_topics(topic_dict, const._TOPIC_FILE)
    '''   
    lock = threading.Lock()
    save_thread = Save(collection = collection, 
                       interval   = const._SAVE_INTERVAL,
                       lock       = lock, 
                       save_dir   = const._CORPUS_FOLDER,
                       mod_num    = const._NUM_RESULT_FOLDERS)
    
    save_thread.start()
    
    while True:       
        try:
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.exchange_declare(exchange=const._EXCHANGE_NAME, 
                                     exchange_type='direct')
          
            channel.queue_declare(queue='new_topics')
            channel.queue_declare(queue='delete_topics')
            #channel.queue_declare(queue='update_topics')   
            channel.queue_bind(exchange=const._EXCHANGE_NAME, 
                               queue='new_topics', routing_key='new')
            channel.queue_bind(exchange=const._EXCHANGE_NAME, 
                               queue='delete_topics', routing_key='delete')
            #channel.queue_bind(exchange=const._EXCHANGE_NAME, queue='update_topics', routing_key='update')
            def on_new_topic(ch, method, properties, body):
                while type(body) != dict:
                    body = json.loads(body)
                
                new_topic = body
                new_topic['postDate'] /= const._TIMESTAMP_FACTOR
                logger.info('Received new topic, id=%s', new_topic['topicID'])

                with lock:
                    collection.add_one(new_topic)
                    collection.remove_old()
                channel.basic_ack(delivery_tag=method.delivery_tag)      

            def on_delete(ch, method, properties, body):
                while type(body) != dict:
                    body = json.loads(body)

                delete_topic = body
                logger.info('Deleting topic %s', delete_topic['topicID'])
                with lock:
                    collection.delete_one(delete_topic['topicID'])
                channel.basic_ack(delivery_tag=method.delivery_tag)
            
            channel.basic_consume(on_new_topic, queue='new_topics')
            channel.basic_consume(on_delete, queue='delete_topics')
            '''
            channel.basic_consume(on_update_topic, queue='update_topics')                                  
            '''    
            logger.info(' [*] Waiting for messages. To exit press CTRL+C')
            channel.start_consuming()
        
        except Exception as e:
            logger.exception(e)
            time.sleep(const._SLEEP_TIME)

    '''
    word_weights = tp.compute_profiles(topic_ids=topic_ids,  
                                       filter_fn=utils.is_valid_text,
                                       features=const._REPLY_FEATURES, 
                                       weights=const._WEIGHTS, 
                                       preprocess_fn=utils.preprocess, 
                                       stopwords=stopwords, 
                                       update=True, 
                                       path=const._PROFILES, 
                                       alpha=args.alpha, 
                                       smartirs=args.smartirs)

    # get k most representative words for each topic
    profile_words = tp.get_profile_words(topic_ids=topic_ids, 
                                         profiles=word_weights,
                                         k=args.k, 
                                         update=True, 
                                         path=const._PROFILE_WORDS)
   
    similarities = sim.compute_similarities(corpus_topic_ids=topic_ids, 
                                            update_topic_ids=topic_ids,
                                            preprocess_fn=utils.preprocess, 
                                            stopwords=stopwords, 
                                            profile_words=profile_words, 
                                            coeff=args.beta,
                                            T=const._T,
                                            update=True, 
                                            path=const._SIMILARITIES)
    
'''
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', action='store_true', help='load previously saved corpus and similarity data')
    parser.add_argument('-c', action='store_true', help='load message queue connection configurations from file')   
    args = parser.parse_args()
    main(args)
