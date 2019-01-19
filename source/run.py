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

    def run(self):
        while True:
            time.sleep(self.interval)
            with self.lock: 
                self._save_topics(self.topic_dir, self.mod_num)
                self._save_recoms(self.recom_dir)
                 
    def _save_topics(self, save_dir):
        self.topics.save(save_dir, self.mod_num)

    def _save_recoms(self, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(recom_path)
        for stid in self.recoms:
            with open(os.path.join(save_dir, stid), 'w') as f:
                json.dump(self.recoms[stid], f)


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
                         singles           = const._SINGLES,
                         puncs             = const._PUNCS,  
                         punc_frac_low     = const._PUNC_FRAC_LOW,
                         punc_frac_high    = const._PUNC_FRAC_HIGH, 
                         valid_count       = const._VALID_COUNT, 
                         valid_ratio       = const._VALID_RATIO,
                         stopwords         = stopwords,
                         trigger_days      = const._TRIGGER_DAYS,
                         keep_days         = const._KEEP_DAYS, 
                         T                 = const._T,
                         duplicate_thresh  = const._DUPLICATE_THRESH, 
                         irrelevant_thresh = const._IRRELEVANT_THRESH, 
                         logger            = utils.get_logger(lc._RUN_LOG_NAME+'.topics')
                         )

    specials = classes.Corpus(singles        = const._SINGLES,
                              puncs          = const._PUNCS,                                                            
                              punc_frac_low  = const._PUNC_FRAC_LOW,
                              punc_frac_high = const._PUNC_FRAC_HIGH, 
                              valid_count    = const._VALID_COUNT, 
                              valid_ratio    = const._VALID_RATIO,
                              stopwords      = stopwords,
                              trigger_days   = const._TRIGGER_DAYS,
                              keep_days      = const._KEEP_DAYS, 
                              T              = const._T,
                              logger         = utils.get_logger(lc._RUN_LOG_NAME+'.specials'))

    recoms = defaultdict(list)
    keyword_weight = defaultdict(list)

    # load previously saved corpus and similarity data if possible
    if args.l:
        try:
            topics.load(const._TOPIC_DIR)
        except FileNotFoundError:
            logger.exception('Topic data files not found. New files will be created')
        try:
            specials.load(const._SPECIAL_DIR)
        except FileNotFoundError:
            logger.exception('Special topic data files not found. New files will be created')

        files = os.listdir(const._RECOM_DIR)
        for file in files:
            if not file.isnumeric():
                continue
            path = os.path.join(const._RECOM_DIR, file)
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
        credentials = pika.PlainCredentials(username = mc._USERNAME,
                                            password = mc._PASSWORD)          
        params = pika.ConnectionParameters(host        = mc._HOST, 
                                           credentials = credentials)
    else:
        params = pika.ConnectionParameters(host='localhost')

    lock = threading.Lock()
    save_topics = Save(topics    = topics, 
                       recoms    = recoms,
                       interval  = const._SAVE_INTERVAL,
                       lock      = lock, 
                       topic_dir = topic_dir,
                       recom_dir = recom_dir,
                       mod_num   = const._NUM_RESULT_DIRS)
    
    save_topics.start()
    
    while True:       
        try:
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.basic_qos(prefetch_count=1)
            channel.exchange_declare(exchange=const._EXCHANGE_NAME, 
                                     exchange_type='direct')
          
            channel.queue_declare(queue='new_topics')
            channel.queue_declare(queue='old_topics')
            channel.queue_declare(queue='special_topics')
            channel.queue_declare(queue='delete_topics')
            #channel.queue_declare(queue='update_topics')   
            channel.queue_bind(exchange=const._EXCHANGE_NAME, 
                               queue='new_topics', routing_key='new')
            channel.queue_bind(exchange=const._EXCHANGE_NAME,
                               queue='old_topics', routing_key='old')
            channel.queue_bind(exchange=const._EXCHANGE_NAME,
                               queue='special_topics', routing_key='special')
            channel.queue_bind(exchange=const._EXCHANGE_NAME, 
                               queue='delete_topics', routing_key='delete')
            #channel.queue_bind(exchange=const._EXCHANGE_NAME, queue='update_topics', routing_key='update')
            def decode_to_dict(msg):
                while type(msg) != dict:
                    msg = json.loads(msg)
                return msg

            def get_relevance(stid, tid):
                relevance = 0
                ts = datetime.fromtimestamp(specials.corpus_data[stid]['date'])
                t = datetime.fromtimestamp(topics.corpus_data[tid]['date'])

                bow = topics.dictionary.doc2bow(topics.corpus_data[tid]['body'])
                tfidf_weight = {wid:weight for wid, weight in keyword_weight[stid]}  

                for word_id, freq in bow:
                    word = topics.dictionary[word_id]
                    if word in specials.dictionary.token2id:
                        word_id_in_specials = specials.dictionary.token2id[word]
                        assert specials.dictionary[word_id_in_specials] == word
                        if word_id_in_specials in tfidf_weight:
                            relevance += freq*tfidf_weight[word_id_in_specials]
                
                relevance *= min(1, math.exp(-(ts-t).days/topics.T))

                return relevance

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
                    if topics.add_one(new_topic, 2*const._TOP_NUM):
                        topics.remove_old()
                        for stid in specials.corpus_data:
                            match_val = get_relevance(stid, new_tid)
                            recom_insert(recoms[stid], new_tid, match_val)

                channel.basic_ack(delivery_tag=method.delivery_tag)      

            def on_old_topic(ch, method, properties, body):
                old_topic = decode_to_dict(body)
                old_tid = str(old_topic['topicID'])
                logger.info('Received old topic %s', old_tid)
                channel.basic_ack(delivery_tag=method.delivery_tag)

                with lock:
                    sim_list = topics.find_most_similar(old_topic)

                sim_list = [tid for tid, val in sim_list][:const._TOP_NUM]
                
                channel.basic_publish(exchange=const._EXCHANGE_NAME,
                                      routing_key='old',
                                      body=json.dumps(sim_list))

            def on_special_topic(ch, method, properties, body):
                special_topic = decode_to_dict(body)
                stid = special_topic['topicID']
                special_topic['postDate'] /= const._TIMESTAMP_FACTOR
                logger.info('Received special topic %s', special_topic['topicID'])

                with lock:
                    if specials.add_one(special_topic):
                        specials.remove_old()
                        model = specials.get_tfidf_model(scheme=const._SMARTIRS_SCHEME) 
                        for stid in specials.corpus_data:
                            bow = specials.dictionary.doc2bow(specials.corpus_data[stid]['body'])
                            keyword_weight[stid] = model[bow]
                            keyword_weight[stid].sort(key=lambda x:x[1], reverse=True)
                            print('keyword_weight: ', keyword_weight)
                            del keyword_weight[stid][const._KEYWORD_NUM:]
                            recom_list = [[tid, get_relevance(stid, tid)] for tid in topics.corpus_data]
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
            logger.info('Retrying in %d seconds', const._SLEEP_TIME)
            time.sleep(const._SLEEP_TIME)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', action='store_true', help='load previously saved corpus and similarity data')
    parser.add_argument('-c', action='store_true', help='load message queue connection configurations from file')   
    args = parser.parse_args()
    main(args)
