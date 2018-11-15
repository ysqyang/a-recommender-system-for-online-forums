# -*- coding: utf-8 -*-

import utils
#import topic_profiling as tp
#import similarity as sim
#import argparse
#from gensim import models
import json
import pika
import constants as const
import sys
import os
import logging
import topics
from datetime import datetime

def update(load_data, read_config):   
    logging.basicConfig(filename=const._RUN_LOG_FILE, filemode='w', 
                        level=const._LOG_LEVEL)
    
    # load stopwords
    stopwords = utils.load_stopwords(const._STOPWORD_FILE)
    collection = topics.Topics(puncs             = const._PUNCS, 
                               singles           = const._SINGLES, 
                               stopwords         = stopwords, 
                               punc_frac_low     = const._PUNC_FRAC_LOW,
                               punc_frac_high    = const._PUNC_FRAC_HIGH, 
                               valid_count       = const._VALID_COUNT, 
                               valid_ratio       = const._VALID_RATIO,
                               trigger_days      = const._TRIGGER_DAYS,
                               keep_days         = const._KEEP_DAYS, 
                               T                 = const._T,
                               irrelevant_thresh = const._IRRELEVANT_THRESH)

    collection.get_dictionary()

    # load previously saved corpus and similarity data if possible
    if load_data:
        try:
            collection.load(const._CORPUS_DATA, const._SIMILARITY_MATRIX, 
                            const._SIMILARITY_SORTED)
        except:
            logging.error('Data file not found or corrupted')
            sys.exit()
    
    # establish rabbitmq connection and declare queues
    if read_config:
        if os.path.exists(const._CONFIG_FILE):
            config = utils.get_config(const._CONFIG_FILE)
            sections = config.sections()
            if len(sections) == 0:
                logging.error('Configuration file is empty')
                sys.exit()

            sec = sections[0]
            
            credentials = pika.PlainCredentials(username=config[sec]['username'],
                                                password=config[sec]['password'])
            
            params = pika.ConnectionParameters(host=config[sec]['host'], 
                                               credentials=credentials)
        else:
            logging.error('Configuration file not found')
            sys.exit()
    
    else:
        params = pika.ConnectionParameters(host='localhost')
    
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.exchange_declare(exchange=const._EXCHANGE_NAME, exchange_type='direct')
  
    channel.queue_declare(queue='new_topics')
    channel.queue_declare(queue='delete_topics')
    #channel.queue_declare(queue='update_topics')   
    channel.queue_bind(exchange=const._EXCHANGE_NAME, queue='new_topics', routing_key='new')
    channel.queue_bind(exchange=const._EXCHANGE_NAME, queue='delete_topics', routing_key='delete')
    #channel.queue_bind(exchange=const._EXCHANGE_NAME, queue='update_topics', routing_key='update')
    
    def on_new_topic(ch, method, properties, body):
        while type(body) != dict:
            body = json.loads(body)
        
        new_topic = body
        new_topic['postDate'] /= const._TIMESTAMP_FACTOR
        logging.info('Received new topic, id=%s', new_topic['topicID'])

        status = collection.add_one(new_topic)

        if status:
            #print('after adding: ', collection.oldest, collection.latest, datetime.fromtimestamp(new_topic['postDate']))
            collection.save(const._CORPUS_DATA, const._SIMILARITY_MATRIX, 
                            const._SIMILARITY_SORTED)      

    def on_delete(ch, method, properties, body):
        while type(body) != dict:
            body = json.loads(body)

        delete_topic = body
        logging.info('Deleting topic %s', delete_topic['topicID'])
        delete_id = str(delete_topic['topicID'])
        status = collection.delete_one(delete_id)
        if status:
            #print('after deleting: ', collection.oldest, collection.latest, delete_date)
            collection.save(const._CORPUS_DATA, const._SIMILARITY_MATRIX, 
                            const._SIMILARITY_SORTED) 

    '''
    def on_subject_update(ch, method, properties, body):
        while type(body) != dict:
            body = json.loads(body)

        subject_dict = body
        logging.info('Updating recommendations for subject %s', subject_dict['subID'])
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
    channel.basic_consume(on_new_topic,
                          queue='new_topics',
                          no_ack=True)
    channel.basic_consume(on_delete, 
                          queue='delete_topics',
                          no_ack=True)

    '''
    channel.basic_consume(on_update_topic, 
                          queue='update_topics',
                          no_ack=True)
    '''    
    logging.info(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

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
    

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', action='store_true', help='load previously saved corpus and similarity data')
    parser.add_argument('-c', action='store_true', help='load message queue connection configurations from file')   
    args = parser.parse_args()
    main(args)
'''