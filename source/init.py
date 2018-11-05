# -*- coding: utf-8 -*-

import utils
#import topic_profiling as tp
#import similarity as sim
import constants as const
#import database
import json
import logging
import pika

def main():
    logging.basicConfig(filename=const._INIT_LOG_FILE, filemode='w', level=logging.DEBUG)
    
    config = utils.get_config(const._CONFIG_FILE)
    logging.info('Configuration loaded')
    sections = config.sections()

    if len(sections) == 0:
        logging.error('Configuration file is empty. Exiting...')
        sys.exit()

    sec = sections[0]
    
    credentials = pika.PlainCredentials(username=config[sec]['username'],
                                        password=config[sec]['password'])
    
    params = pika.ConnectionParameters(host=config[sec]['host'], 
                                       credentials=credentials)

    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.exchange_declare(exchange=const._EXCHANGE_NAME, exchange_type='direct')
    channel.queue_declare(queue='all_topics')
    channel.queue_bind(exchange=const._EXCHANGE_NAME, queue='all_topics', routing_key='all')

    path = const._TOPIC_FILE
    topic_dict = {}

    def call_back(ch, method, properties, body):
        logging.info('Received topics')
        topic = json.loads(body)
        topic_id = topic['topicID']
        topic_dict[topic_id] = {k:v for k, v in new_topic.items() if k != 'topicID'} 
        utils.save_topics(topic_dict, path)
       
    channel.basic_consume(call_back,
                          queue='all_topics',
                          no_ack=True)

    logging.info(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()
             
    #utils.load_replies(db, topic_ids, const._REPLY_FEATURES, const._REPLY_FILE)
    '''
    word_weights = tp.compute_profiles(topic_ids=topic_ids,  
                                       features=const._REPLY_FEATURES, 
                                       weights=const._WEIGHTS, 
                                       preprocess_fn=utils.preprocess, 
                                       stopwords=stopwords, 
                                       update=False, 
                                       path=const._PROFILES, 
                                       alpha=args.alpha, 
                                       smartirs=args.smartirs)

    topic_ids = list(word_weights.keys())
    # get k most representative words for each topic
    profile_words = tp.get_profile_words(profiles=word_weights,
                                         k=args.k, 
                                         update=False, 
                                         path=const._PROFILE_WORDS)
   
    similarities = sim.compute_similarities(corpus_topic_ids=topic_ids, 
                                            active_topic_ids=topic_ids,
                                            profile_words=profile_words,
                                            preprocess_fn=utils.preprocess, 
                                            stopwords=stopwords, 
                                            coeff=args.beta,
                                            T=const._T,
                                            update=False, 
                                            path=const._SIMILARITIES)
    
    collection = topics.Topic_collection(const._TOPIC_FILE)
    collection.make_corpus(preprocess_fn=utils.preprocess, 
                           stopwords=stopwords, 
                           punc_frac_low=const._PUNC_FRAC_LOW, 
                           punc_frac_high=const._PUNC_FRAC_HIGH, 
                           valid_ratio=const._VALID_RATIO)
    print('共{}条候选主贴可供推荐'.format(len(collection.valid_topics)))
    collection.get_bow()
    collection.get_similarity_matrix(const._SIMILARITIES, const._T)
    for tid in collection.valid_topics:
        print(topics_all[tid]['body'])
        recoms = collection.generate_recommendations(topic_id=tid, 
                                                     duplicate_thresh=const._DUPLICATE_THRESH,
                                                     irrelevant_thresh=const._IRRELEVANT_THRESH)
        print()
        for i, (tid, score) in enumerate(recoms):
            print('*'*30+'推荐文章{}:'.format(i+1)+'*'*30, end='  ')
            print('相似指数:', score)
            print(topics_all[tid]['body'])
            print()
        print('-'*150)
        print('-'*150)
    '''
if __name__ == '__main__': 
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--alpha', type=float, default=0.7, 
                        #help='''contribution coefficient for topic content 
                                #in computing word weights''')
    #parser.add_argument('--k', type=int, default=10, 
                        #help='number of words to represent a discussion thread')
    #parser.add_argument('--beta', type=float, default=0.8,
                        #help='''contribution coefficient for in-document frequency
                                #in computing word probabilities''')
    #parser.add_argument('--smartirs', type=str, default='atn', help='type of tf-idf variants')

    #args = parser.parse_args()
    main()
