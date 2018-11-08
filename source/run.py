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
from datetime import timedelta

def main():   
    logging.basicConfig(filename=const._RUN_LOG_FILE, filemode='w', level=logging.DEBUG)
    
    if const._TRIGGER_DAYS <= const._KEEP_DAYS:
        error_msg = 'Trigger_days must be strictly greater than keep_days'
        logging.error(error_msg)
        raise ValueError(error_msg)

    # load stopwords
    stopwords = utils.load_stopwords(const._STOPWORD_FILE)
    collection = topics.Topic_collection(puncs=const._PUNCS, 
                                         singles=const._SINGLES, 
                                         stopwords=stopwords, 
                                         punc_frac_low=const._PUNC_FRAC_LOW,
                                         punc_frac_high=const._PUNC_FRAC_HIGH, 
                                         valid_count=const._VALID_COUNT, 
                                         valid_ratio=const._VALID_RATIO, 
                                         T=const._T)

    # load previously saved corpus and similarity data if possible
    if os.path.exists(const._CORPUS_DATA)           \
       and os.path.exists(const._SIMILARITY_MATRIX) \
       and os.path.exists(const._SIMILARITY_SORTED):
        collection.load(const._CORPUS_DATA, const._SIMILARITY_MATRIX, const._SIMILARITY_SORTED)
    
    '''
    # establish rabbitmq connection and declare queues
    config = utils.get_config(const._CONFIG_FILE)
    sections = config.sections()

    if len(sections) == 0:
        logging.error('Configuration file is empty. Exiting...')
        sys.exit()

    sec = sections[0]
    
    credentials = pika.PlainCredentials(username=config[sec]['username'],
                                        password=config[sec]['password'])
    
    params = pika.ConnectionParameters(host=config[sec]['host'], 
                                       credentials=credentials)
    
    '''
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
        logging.info('Received new topic')
        #print(body, type(body))
        new_topic = json.loads(json.loads(body))
        print(new_topic, type(new_topic))

        collection.add_one(new_topic)

        oldest, latest = collection.get_oldest(), collection.get_latest()
        if oldest and latest and (latest-oldest).days > const._TRIGGER_DAYS:
            cut_off = latest - const._KEEP_DAYS
            collection.remove_old(cut_off)

        collection.save(const._CORPUS_DATA, const._SIMILARITY_MATRIX, 
                        const._SIMILARITY_SORTED) 
        #print('sorted tids after adding new topic:', tids)      

    def on_delete(ch, method, properties, body):
        logging.info('Received topic to be deleted')
        delete_id = json.loads(body)
        if delete_id not in collection.corpus_data:
            logging.warning('Topic %s not found in the collection', delete_id)
            return

        collection.delete_one(delete_id)
        collection.save(const._CORPUS_DATA, const._SIMILARITY_MATRIX, 
                        const._SIMILARITY_SORTED) 

    '''
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

    '''

if __name__ == '__main__': 
    #parser = argparse.ArgumentParser()
    
    #parser.add_argument('--alpha', type=float, default=0.7, 
    #                     help='''contribution coefficient for topic content 
    #                            in computing word weights''')
    #parser.add_argument('--k', type=int, default=60, 
    #                    help='number of words to represent a discussion thread')
    #parser.add_argument('--beta', type=float, default=0.5,
    #                    help='''contribution coefficient for in-document frequency
    #                            in computing word probabilities''')
    #parser.add_argument('--T', type=float, default=365, help='time attenuation factor')
    #parser.add_argument('--smartirs', type=str, default='atn', help='type of tf-idf variants')

    #args = parser.parse_args()
    main()