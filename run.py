import utils
import topic_profiling as tp
import similarity as sim
import argparse
from gensim import models
import json
import pika
import constants as const

def main(args):
    with open(const._TOPIC_FILE, 'r') as f:
        topics = json.load(f)

    collection = topics.Topic_collection(const._TOPIC_FILE)
    collection.make_corpus(preprocess_fn=utils.preprocess, 
                           stopwords=stopwords, 
                           punc_frac_low=const._PUNC_FRAC_LOW, 
                           punc_frac_high=const._PUNC_FRAC_HIGH, 
                           valid_ratio=const._VALID_RATIO)
    print('共{}条候选主贴可供推荐'.format(len(collection.valid_topics)))
    collection.get_bow()
    collection.get_similarity_matrix(const._SIMILARITIES, const._T)

    stopwords = utils.load_stopwords(const._STOPWORD_FILE)

    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='new_topics')
    channel.queue_declare(queue='active_topics')
    channel.queue_declare(queue='topics_to_delete')

    def on_new_topics(ch, method, properties, body):
        new_topic = json.loads(body)
        topic_id = new_topic['topicid']
        topics[topic_id] = {k:v for k, v in new_topic.items() if k != 'topicid'}
        collection.update(topic=new_topic,
        	              preprocess_fn=utils.preprocess, 
        	              stopwords=stopwords,
        	              punc_frac_low=const._PUNC_FRAC_LOW, 
        	              punc_frac_high=const._PUNC_FRAC_HIGH, 
        	              valid_ratio=const._VALID_RATIO)

    def on_active_topics(ch, method, properties, body):
    	active_topic = json.loads(body)
		topic_id = active_topic['topicid']
    	topics[]


    channel.basic_consume(on_new_topics,
                          queue='new_topics',
                          no_ack=True)

    channel.basic_consume(on_active_topic, 
    					  queue='active_topics',
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
                                            active_topic_ids=topic_ids,
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
    
    #parser.add_argument('--alpha', type=float, default=0.7, 
    #                     help='''contribution coefficient for topic content 
    #                            in computing word weights''')
    #parser.add_argument('--k', type=int, default=60, 
    #                    help='number of words to represent a discussion thread')
    #parser.add_argument('--beta', type=float, default=0.5,
    #                    help='''contribution coefficient for in-document frequency
    #                            in computing word probabilities''')
    parser.add_argument('--T', type=float, default=365, help='time attenuation factor')
    #parser.add_argument('--smartirs', type=str, default='atn', help='type of tf-idf variants')

    args = parser.parse_args()
    main(args)