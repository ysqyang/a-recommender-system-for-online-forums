import utils
#import topic_profiling as tp
#import similarity as sim
from gensim.models import TfidfModel
import json
import math
import pika
import os, sys
import classes
from datetime import datetime
import time
from collections import defaultdict
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
    def __init__(self, topics, interval, lock, save_dir, mod_num, logger=None):
        threading.Thread.__init__(self)
        self.topics = topics
        self.interval = interval
        self.lock = lock
        self.save_dir = save_dir
        self.mod_num = mod_num

    def run(self):
        while True:
            time.sleep(self.interval)
            with self.lock:
                self.topics.save(self.save_dir, self.mod_num)
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

    topics = classes.Corpus_with_similarity_data( 
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
                         duplicate_thresh  = const._DUPLICATE_THRESH, 
                         irrelevant_thresh = const._IRRELEVANT_THRESH, 
                         max_size          = 2*const._TOP_NUM,
                         logger            = utils.get_logger(lc._RUN_LOG_NAME+'.topics')
                         )

    specials = classes.Corpus(singles      = const._SINGLES, 
                              stopwords    = stopwords, 
                              trigger_days = const._TRIGGER_DAYS,
                              keep_days    = const._KEEP_DAYS, 
                              T            = const._T,
                              logger       = utils.get_logger(lc._RUN_LOG_NAME+'.specials'))

    topics.get_dictionary()
    specials.get_dictionary()
    recoms = defaultdict(list)
    keyword_weight = defaultdict(list)

    # load previously saved corpus and similarity data if possible
    if args.l:
        try:
            topics.load(const._CORPUS_DIR)
        except FileNotFoundError:
            logger.exception('Topic data files not found. New files will be created')
        try:
            specials.load(const._SPECIAL_CORPUS_DIR)
        except FileNotFoundError:
            logger.exception('Special topic data files not found. New files will be created')
        try:
            recoms = json.load(const._RECOMS)
        except FileNotFoundError:
            logger.exception('Recommendation data file not found. New dict will be instantiated')
        except json.JSONDecodeError:
            logger.exception('Failed to load recommendation data. New dict will be instantiated')
    
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
    
    lock = threading.Lock()
    save_topics = Save(topics = topics, 
                       interval   = const._SAVE_INTERVAL,
                       lock       = lock, 
                       save_dir   = const._CORPUS_DIR,
                       mod_num    = const._NUM_RESULT_DIRS)
    
    save_topics.start()
    
    while True:       
        try:
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.exchange_declare(exchange=const._EXCHANGE_NAME, 
                                     exchange_type='direct')
          
            channel.queue_declare(queue='new_topics')
            channel.queue_declare(queue='special_topics')
            channel.queue_declare(queue='delete_topics')
            #channel.queue_declare(queue='update_topics')   
            channel.queue_bind(exchange=const._EXCHANGE_NAME, 
                               queue='new_topics', routing_key='new')
            channel.queue_bind(exchange=const._EXCHANGE_NAME,
                               queue='special_topics', routing_key='special')
            channel.queue_bind(exchange=const._EXCHANGE_NAME, 
                               queue='delete_topics', routing_key='delete')
            #channel.queue_bind(exchange=const._EXCHANGE_NAME, queue='update_topics', routing_key='update')
            def decode_to_dict(msg):
                while type(msg) != dict:
                    msg = json.loads(msg)
                return msg

            '''
            def get_relevance(stid, tid):
                relevance = 0
                ts = datetime.fromtimestamp(specials.corpus_data[stid]['date'])
                t = datetime.fromtimestamp(topics.corpus_data[tid]['date'])

                bow = topics.corpus_data[topic_id]['bow']
                tfidf_weight = keyword_weight[stid]  

                for word_id, freq in bow:
                    word = dictionary[word_id]
                    if word in keyword_weight:
                        relevance += freq*keyword_weight[word]
                
                relevance *= math.min(1, math.exp(-(ts-t).days/self.T))

                return relevance
            '''
            
            def on_new_topic(ch, method, properties, body):
                new_topic = decode_to_dict(body)
                new_topic['postDate'] /= const._TIMESTAMP_FACTOR
                new_tid = str(new_topic['topicID'])
                logger.info('Received new topic %s', new_tid)

                def recom_insert(target_list, tid, match_val):
                    i = 0
                    while i < len(target_list) and target_list[i][1] > match_val:
                        i += 1
                    target_list.insert(i, [tid, match_val])

                with lock:
                    if topics.add_one(new_topic):
                        topics.remove_old()
                        '''
                        for stid in specials.corpus_data:
                            match_val = get_relevance(stid, new_tid)
                            recom_insert(recoms[stid], new_tid, match_val)
                        '''
                channel.basic_ack(delivery_tag=method.delivery_tag)      

            def on_special_topic(ch, method, properties, body):
                special_topic = decode_to_dict(body)
                stid = special_topic['topicID']
                special_topic['postDate'] /= const._TIMESTAMP_FACTOR
                logger.info('Received special topic %s', special_topic['topicID'])

                with lock:
                    if specials.add_one(special_topic):
                        specials.remove_old()
                        model = specials.get_tfidf_model() 
                        for stid in specials.corpus_data:
                            keyword_weight[stid] = model[specials.corpus_data[stid]['bow']]
                            keyword_weight[stid].sort(key=lambda x:x[1], reverse=True)
                            recom_list = [[tid, get_relevance(stid, tid)] for tid in topics]
                            recom_list.sort(key=lambda x:x[1], reverse=True)
                            recoms[stid] = recom_list
                
                channel.basic_ack(delivery_tag=method.delivery_tag) 

            def on_delete(ch, method, properties, body):
                delete_topic = decode_to_dict(body)
                logger.info('Deleting topic %s', delete_topic['topicID'])
                
                with lock:
                    topics.delete_one(delete_topic['topicID'])
                channel.basic_ack(delivery_tag=method.delivery_tag)
            
            '''
            def on_update_topic(ch, method, properties, body):
                update_topic = json.loads(body)
                topic_id = update_topic['topicID']
                for attr in update_topic:
                    if attr != 'topicID':
                        topic_dict[topic_id][attr] = update_topic[attr]
                utils.save_topics(topic_dict, const._TOPIC_FILE)
            '''  

            channel.basic_consume(on_new_topic, queue='new_topics')
            channel.basic_consume(on_special_topic, queue='special_topics')
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
