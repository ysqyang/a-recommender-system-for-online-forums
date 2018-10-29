import utils
import topic_profiling as tp
import similarity as sim
import argparse
#from gensim import models
import json
import pika
import constants as const
import configparser
import sys
import topics
from datetime import datetime, timedelta

def main():   
    with open(const._TOPIC_FILE, 'r') as f:
        topic_dict = json.load(f)

    tids = sorted(list(topic_dict.keys()))
    stopwords = utils.load_stopwords(const._STOPWORD_FILE)
    collection = topics.Topic_collection(topic_dict, const._DATETIME_FORMAT)
    collection.make_corpus(preprocess_fn=utils.preprocess, 
                           stopwords=stopwords, 
                           punc_frac_low=const._PUNC_FRAC_LOW, 
                           punc_frac_high=const._PUNC_FRAC_HIGH,
                           valid_count=const._VALID_COUNT, 
                           valid_ratio=const._VALID_RATIO)

    print('共{}条候选主贴可供推荐'.format(len(collection.valid_topics)))
    collection.get_bow()
    collection.get_similarity_data(const._T)
    collection.save_similarity_data(const._SIMILARITY_MATRIX, const._SIMILARITY_SORTED)

    print('dates: ', collection.dates)

    '''
    config = utils.get_config(const._CONFIG_FILE)
    sections = config.sections()

    if len(sections) == 0:
        print('Configuration file is empty.')
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
        print('receiving new topic...')
        new_topic = json.loads(body)
        topic_id = new_topic['topicid']

        oldest = topic_dict[tids[0]]['POSTDATE']
        oldest = datetime.strptime(oldest, const._DATETIME_FORMAT)
        print('oldest: ', oldest)
        post_date = datetime.strptime(new_topic['POSTDATE'], const._DATETIME_FORMAT)
        cut_off = post_date - timedelta(days=const._KEEP_DAYS)
        
        def remove_old(cut_off):
            if cut_off < oldest or cut_off > post_date:
                return 
            
            for tid in tids:
                dt = datetime.strptime(topic_dict[tid]['POSTDATE'], const._DATETIME_FORMAT)
                if dt.date() >= cut_off.date():
                    break  
                del topic_dict[tid]

        if (post_date - oldest).days > const._TRIGGER_DAYS:
            remove_old(cut_off)

        topic_dict[topic_id] = {k:v for k, v in new_topic.items() if k != 'topicid'}
        utils.save_topics(topic_dict, const._TMP)

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

        collection.save_similarity_data(const._SIMILARITY_MATRIX, const._SIMILARITY_SORTED)
        
        for tid, date in zip(collection.valid_topics, collection.dates):
            print(tid, date)

        print('corpus size: ', len(collection.corpus), len(collection.corpus_bow), 
         len(collection.dates), len(collection.valid_topics), len(collection.sim_matrix))

    def on_update_topic(ch, method, properties, body):
        update_topic = json.loads(body)
        topic_id = update_topic['topicid']
        for attr in update_topic:
            if attr != 'topicid':
                topic_dict[topic_id][attr] = update_topic[attr]
        utils.save_topics(topic_dict, const._TMP)

    def on_delete(ch, method, properties, body):
        print('deleting topic...')
        delete_id = json.loads(body)
        if delete_id not in topic_dict:
            return

        print('HERE!!!!!')
        del topic_dict[delete_id]
        utils.save_topics(topic_dict, const._TMP)

        collection.delete_one(delete_id)
        collection.save_similarity_data(const._SIMILARITY_MATRIX, const._SIMILARITY_SORTED)

        for tid, date in zip(collection.valid_topics, collection.dates):
            print(tid, date)

        print('corpus size: ', len(collection.corpus), len(collection.corpus_bow), 
         len(collection.dates), len(collection.valid_topics), len(collection.sim_matrix))
        

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