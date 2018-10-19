import utils
import topic_profiling as tp
import similarity as sim
import argparse
from gensim import models
import json
import pika

def main(args):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='active_topics')

    def callback(ch, method, properties, body):
        print(" [x] Received {}".format(body))

    channel.basic_consume(callback,
                          queue='active_topics',
                          no_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()



    stopwords = utils.load_stopwords(const._STOPWORDS)
    
    with open(const._TOPIC_FILE, 'r') as f1, open(const._REPLY_FILE, 'r') as f2:
        topics, replies = json.load(f1), json.load(f2)

    for  


    word_weights = tp.compute_profiles(topic_ids=args.active_topics,  
                                       features=const._FEATURES, 
                                       weights=const._WEIGHTS, 
                                       preprocess_fn=utils.preprocess, 
                                       stopwords=stopwords, 
                                       update=True, 
                                       path=const._WORD_WEIGHTS, 
                                       alpha=args.alpha, 
                                       smartirs=args.smartirs)

    # get k most representative words for each topic
    profile_words = tp.get_profile_words(topic_ids=topic_ids, 
                                         word_weights=word_weights,
                                         k=args.k, 
                                         update=True, 
                                         path=const._PROFILE_WORDS)

    similarity_all = sim.get_similarity_all(db, utils.preprocess, 
                     stopwords, profile_words, args.beta)
    # save computed similarity data to file
    with open(const._SAVE_PATH_SIMILARITY, 'wb') as f:
        pickle.dump(similarity_all, f)

    print('similarity matrices computed and saved to disk')

    adjust_for_time(tid_to_date, similarity_all, args.T) 

    with open(const._SAVE_PATH_SIMILARITY_ADJUSTED, 'wb') as f:
        pickle.dump(similarity_all, f)

    print('adjusted similarity matrices computed and saved to disk')

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.7, 
                        help='''contribution coefficient for topic content 
                                in computing word weights''')
    parser.add_argument('--k', type=int, default=60, 
                        help='number of words to represent a discussion thread')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='''contribution coefficient for in-document frequency
                                in computing word probabilities''')
    parser.add_argument('--T', type=float, default=365, help='time attenuation factor')
    parser.add_argument('--smartirs', type=str, default='atn', help='type of tf-idf variants')

    args = parser.parse_args()
    main(args)