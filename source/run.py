import utils
#import topic_profiling as tp
#import similarity as sim
#import argparse
#from gensim import models
import json
import pika
import constants as const
import sys
import logging
import topics
from datetime import datetime, timedelta

def main():   
    logging.basicConfig(filename=const._RUN_LOG_FILE, filemode='w', level=logging.DEBUG)

    with open(const._TOPIC_FILE, 'r') as f:
        topic_dict = json.load(f)

    sorted_tids = sorted(list(topic_dict.keys()))
    stopwords = utils.load_stopwords(const._STOPWORD_FILE)   

    collection = topics.Topic_collection(topic_dict, const._DATETIME_FORMAT)
    collection.get_corpus_data(preprocess_fn=utils.preprocess, 
                               stopwords=stopwords, 
                               punc_frac_low=const._PUNC_FRAC_LOW, 
                               punc_frac_high=const._PUNC_FRAC_HIGH,
                               valid_count=const._VALID_COUNT, 
                               valid_ratio=const._VALID_RATIO)

    collection.get_similarity_data(const._T)

    collection.save_similarity_data(const._SIMILARITY_MATRIX, const._SIMILARITY_SORTED)
    '''
    config = utils.get_config(const._CONFIG_FILE)
    logging.info('Configuration loaded')
    sections = config.sections()

    if len(sections) == 0:
        logging.error('Configuration file is empty. Exiting...')
        sys.exit()

    sec = sections[0]
    
    credentials = pika.PlainCredentials(username=config[sec]['username'],
                                        password=config[sec]['password'])
    
    params = pika.connectionParameters(host=config[sec]['host'], 
                                       credentials=credentials)
    '''
    params = pika.ConnectionParameters(host='localhost')
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.exchange_declare(exchange=const._EXCHANGE_NAME, exchange_type='direct')

    channel.queue_declare(queue='new_topics')
    channel.queue_declare(queue='update_topics')
    channel.queue_declare(queue='delete_topics')
    channel.queue_bind(exchange=const._EXCHANGE_NAME, queue='new_topics', routing_key='new')
    channel.queue_bind(exchange=const._EXCHANGE_NAME, queue='update_topics', routing_key='update')
    channel.queue_bind(exchange=const._EXCHANGE_NAME, queue='delete_topics', routing_key='delete')

    def on_new_topic(ch, method, properties, body):
        logging.info('Received new topic')
        new_topic = json.loads(body)
        topic_id = new_topic['topicid']
        if topic_id in topic_dict:
            logging.warning('Topic id already exists! Quitting handler...')
            return

        oldest, latest = topic_dict[sorted_tids[0]]['POSTDATE'], new_topic['POSTDATE']
        logging.debug('oldest=%s, latest=%s', oldest, latest)
        oldest = datetime.strptime(oldest, const._DATETIME_FORMAT)
        latest = datetime.strptime(latest, const._DATETIME_FORMAT)
        cut_off = latest - timedelta(days=const._KEEP_DAYS)

        topic_dict[topic_id] = {k:v for k, v in new_topic.items() if k != 'topicid'}
        sorted_tids.append(topic_id)
        
        def remove_old(tids, cut_off):
            if cut_off <= oldest or cut_off > latest:
                return 
            
            logging.info('Deleting old topics')
            for i, tid in enumerate(tids):
                dt = datetime.strptime(topic_dict[tid]['POSTDATE'], const._DATETIME_FORMAT)
                if dt.date() >= cut_off.date():
                    break  
                del topic_dict[tid]
                
            last_cut = topic_dict[tids[i-1]]['POSTDATE']
            new_oldest = topic_dict[tids[i]]['POSTDATE']
            logging.debug('oldest in topic file after removing: %s', new_oldest)
            logging.debug('latest among deleted topics: %s', last_cut)
            new_oldest = datetime.strptime(new_oldest, const._DATETIME_FORMAT)
            last_cut = datetime.strptime(last_cut, const._DATETIME_FORMAT)
            assert last_cut < cut_off <= new_oldest
            del tids[:i]
            assert len(tids) == len(topic_dict) > 0

        if (latest - oldest).days > const._TRIGGER_DAYS:           
            remove_old(sorted_tids, cut_off)
            print(sorted_tids)
         
        utils.save_topics(topic_dict, const._TMP)
        logging.info('New topic added to local disk')

        collection.add_one(topic=new_topic,
                           preprocess_fn=utils.preprocess, 
                           stopwords=stopwords,
                           punc_frac_low=const._PUNC_FRAC_LOW, 
                           punc_frac_high=const._PUNC_FRAC_HIGH, 
                           valid_count=const._VALID_COUNT,
                           valid_ratio=const._VALID_RATIO, 
                           trigger_days=const._TRIGGER_DAYS,
                           cut_off=cut_off, 
                           T=const._T)

        oldest = collection.corpus_data[0]['date']
        oldest = datetime.strptime(oldest, const._DATETIME_FORMAT)
        if (latest - oldest).days > const._TRIGGER_DAYS:
            collection.remove_old(cut_off)
            print(sorted([t['topic_id'] for t in collection.corpus_data]))

        collection.save_similarity_data(const._SIMILARITY_MATRIX, const._SIMILARITY_SORTED)

        #print('sorted tids after adding new topic:', tids)      

    def on_update_topic(ch, method, properties, body):
        update_topic = json.loads(body)
        topic_id = update_topic['topicid']
        for attr in update_topic:
            if attr != 'topicid':
                topic_dict[topic_id][attr] = update_topic[attr]
        utils.save_topics(topic_dict, const._TMP)

    def on_delete(ch, method, properties, body):
        logging.info('Received topic to be deleted')
        delete_id = json.loads(body)
        if delete_id not in topic_dict:
            logging.warning('Topic not found in the collection')
            return

        del topic_dict[delete_id]
        sorted_tids.remove(delete_id)
        assert delete_id not in sorted_tids and delete_id not in topic_dict
        utils.save_topics(topic_dict, const._TMP)
        logging.info('Topic deleted from local disk')

        collection.delete_one(delete_id)

        collection.save_similarity_data(const._SIMILARITY_MATRIX, const._SIMILARITY_SORTED)    

    channel.basic_consume(on_new_topic,
                          queue='new_topics',
                          no_ack=True)

    channel.basic_consume(on_update_topic, 
                          queue='update_topics',
                          no_ack=True)

    channel.basic_consume(on_delete, 
                          queue='delete_topics',
                          no_ack=True)
    
    print(' [*] Waiting for messages. To exit press CTRL+C')
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