import utils
import topic_profiling as tp
import similarity as sim
import argparse
from gensim import models
import constants as const
import os

def main(args):
    '''
    db = utils.get_database(_DB_INFO)
    if os.path.exists(const._TOPIC_ID_TO_TABLE_NUM):
        with open(const._TOPIC_ID_TO_TABLE_NUM, 'rb') as f:
            existing = pickle.load(f).keys() 
        new_topics = get_new_topics(db, existing)
        tid_to_table = utils.update_tid_to_table_num_mapping(
                             const._TOPIC_ID_TO_TABLE_NUM, db, new_topics)
    else:
        tid_to_table = utils.create_topic_id_to_table_num(
                             db, const._TOPIC_ID_TO_TABLE_NUM)
    
    if os.path.exists(const._TOPIC_ID_TO_REPLY_TABLE_NUM):
        tid_to_reply_table = utils.update_tid_to_reply_table_num_mapping(
                                   const._TOPIC_ID_TO_REPLY_TABLE_NUM, db, new_topics)
    else:
        tid_to_reply_table = utils.create_topic_id_to_reply_table(
                                   db, tid_to_table.keys(), const._TOPIC_ID_TO_REPLY_TABLE_NUM)

    if os.path.exists(const._TOPIC_ID_TO_DATE):
        tid_to_date = utils.update_tid_to_date_mapping(
                            const._TOPIC_ID_TO_DATE, db, new_topics, tid_to_table)
    else:
        tid_to_date = utils.create_topic_id_to_date(db, const._TOPIC_ID_TO_DATE)
    '''
    stopwords = utils.load_stopwords(const._STOPWORDS)
    db = utils.get_database(const._DB_INFO)
    tid_to_table = utils.load_mapping(const._TOPIC_ID_TO_TABLE_NUM)
    tid_to_reply_table = utils.load_mapping(const._TOPIC_ID_TO_REPLY_TABLE_NUM)
    tid_to_date = utils.load_mapping(const._TOPIC_ID_TO_DATE)

    topic_ids = list(tid_to_table.keys())[:5]
    
    word_weights = tp.compute_profiles(db=db, 
                                       topic_ids=topic_ids, 
                                       tid_to_table=tid_to_table,
                                       tid_to_reply_table=tid_to_reply_table, 
                                       features=const._FEATURES, 
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
  
    similarities = sim.compute_similarities(db=db,
                                            topic_ids=topic_ids, 
                                            preprocess_fn=utils.preprocess, 
                                            stopwords=stopwords, 
                                            profile_words=profile_words, 
                                            coeff=args.beta, 
                                            update=False, 
                                            path=const._SIMILARITIES)

    sim.adjust_for_time(tid_to_date=tid_to_date, 
                        similarities=similarities, 
                        T=args.T, 
                        path=const._SIMILARITIES_ADJUSTED) 

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
