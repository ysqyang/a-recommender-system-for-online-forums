import utils
import topic_profiling as tp
import similarity as sim
import argparse
import constants as const
import topics
import database

def main(args):
    stopwords = utils.load_stopwords(const._STOPWORDS)
    db = database.Database(*const._DB_INFO)
    topic_ids = utils.load_topics(db, const._TOPIC_FEATURES, const._DAYS, 
                                  const._MIN_LEN, const._MIN_REPLIES, 
                                  const._MIN_REPLIES_1, const._TOPIC_FILE)

    utils.load_replies(db, topic_ids, const._REPLY_FEATURES, const._REPLY_FILE)
    word_weights = tp.compute_profiles(topic_ids=topic_ids,  
                                       features=const._REPLY_FEATURES, 
                                       weights=const._WEIGHTS, 
                                       preprocess_fn=utils.preprocess, 
                                       stopwords=stopwords, 
                                       update=False, 
                                       path=const._PROFILES, 
                                       alpha=args.alpha, 
                                       smartirs=args.smartirs)

    # get k most representative words for each topic
    profile_words = tp.get_profile_words(topic_ids=topic_ids, 
                                         profiles=word_weights,
                                         k=args.k, 
                                         update=False, 
                                         path=const._PROFILE_WORDS)
   
    similarities = sim.compute_similarities(corpus_topic_ids=topic_ids, 
                                            active_topic_ids=topic_ids,
                                            preprocess_fn=utils.preprocess, 
                                            stopwords=stopwords, 
                                            profile_words=profile_words, 
                                            coeff=args.beta,
                                            T=const._T,
                                            update=False, 
                                            path=const._SIMILARITIES)
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.7, 
                        help='''contribution coefficient for topic content 
                                in computing word weights''')
    parser.add_argument('--k', type=int, default=10, 
                        help='number of words to represent a discussion thread')
    parser.add_argument('--beta', type=float, default=0.7,
                        help='''contribution coefficient for in-document frequency
                                in computing word probabilities''')
    parser.add_argument('--smartirs', type=str, default='atn', help='type of tf-idf variants')

    args = parser.parse_args()
    main(args)
