import utils
import topic_profiling as tp
import similarity as sim
import argparse
import constants as const
import topics
import database
import json

def main(args):
    stopwords = utils.load_stopwords(const._STOPWORDS)
    '''
    db = database.Database(*const._DB_INFO)
    topic_ids = utils.load_topics(db, const._TOPIC_FEATURES, const._DAYS, 
                                  const._MIN_LEN, const._MIN_REPLIES, 
                                  const._MIN_REPLIES_1, const._TOPIC_FILE)

    utils.load_replies(db, topic_ids, const._REPLY_FEATURES, const._REPLY_FILE)
    '''
    with open(const._TOPIC_FILE, 'r') as f1, open(const._REPLY_FILE, 'r') as f2:
        topics_all, replies_all = json.load(f1), json.load(f2)
    topic_ids = list(topics_all.keys())
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
    '''
    collection = topics.Topic_collection(topic_ids)
    collection.make_corpus(preprocess_fn=utils.preprocess, 
                           stopwords=stopwords, 
                           punc_frac_low=const._PUNC_FRAC_LOW, 
                           punc_frac_high=const._PUNC_FRAC_HIGH, 
                           valid_ratio=const._VALID_RATIO)
    print('共{}条候选可推荐主贴'.format(len(collection.valid_topics)))
    collection.get_bow()
    collection.get_similarity_matrix(const._SIMILARITIES, const._T)
    for tid in ['1506404', '1506413']:
        print(topics_all[tid]['body'])
        recoms = collection.generate_recommendations(topic_id=tid, 
                                                     sim_thresh_low=const._SIM_THRESH_LOW, 
                                                     sim_thresh_high=const._SIM_THRESH_HIGH)
        print('recommendations for topic {}:'.format(tid))
        for tid, score in recoms:
            print(topics_all[tid]['body'], score)
            print('-'*50)
        print('*'*80)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.7, 
                        help='''contribution coefficient for topic content 
                                in computing word weights''')
    parser.add_argument('--k', type=int, default=10, 
                        help='number of words to represent a discussion thread')
    parser.add_argument('--beta', type=float, default=0.8,
                        help='''contribution coefficient for in-document frequency
                                in computing word probabilities''')
    parser.add_argument('--smartirs', type=str, default='atn', help='type of tf-idf variants')

    args = parser.parse_args()
    main(args)
