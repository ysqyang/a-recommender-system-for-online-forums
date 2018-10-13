import utils
import topic_profiling as tp
import similarity
import argparse
from gensim import models
import constants as const

def main(args):
    '''
    db = utils.get_database(_DB_INFO)
    print('connection to database established')
    tid_to_table = ut.create_topic_id_to_table_num(db, _TOPIC_ID_TO_TABLE_NUM)
    print('topic-id-to-table-number mapping created', len(tid_to_table))
    tid_to_reply_table = ut.create_topic_id_to_reply_table(
                         db, tid_to_table.keys(), _TOPIC_ID_TO_REPLY_TABLE_NUM)
    print('topic-id-to-reply-table-number mapping created')
    tid_to_date = ut.create_topic_id_to_date(db, _TOPIC_ID_TO_DATE)
    print('topic-id-to-post-date mapping created')
    '''
    stopwords = utils.load_stopwords(const._STOPWORDS)
    db = utils.get_database(const._DB_INFO)
    tid_to_table = utils.load_mapping(const._TOPIC_ID_TO_TABLE_NUM)
    tid_to_reply_table = utils.load_mapping(const._TOPIC_ID_TO_REPLY_TABLE_NUM)
    tid_to_date = utils.load_mapping(const._TOPIC_ID_TO_DATE)

    topic_ids = list(tid_to_table.keys())
    
    word_weights = tp.compute_profiles(db=db, 
                                       topic_ids=topic_ids, 
                                       tid_to_table=tid_to_table,
                                       tid_to_reply_table=tid_to_reply_table, 
                                       features=const._FEATURES, 
                                       weights=const._WEIGHTS, 
                                       preprocess_fn=utils.preprocess, 
                                       stopwords=stopwords, 
                                       update=False, 
                                       path=const._WORD_WEIGHTS, 
                                       alpha=args.alpha, 
                                       smartirs=args.smartirs)

    # get k most representative words for each topic
    profile_words = tp.get_profile_words(topic_ids=topic_ids, 
                                         word_weights=word_weights,
                                         k=args.k, 
                                         update=False, 
                                         path=const._PROFILE_WORDS)

    
    
    similarity_all = similarity.get_similarity_all(db, utils.preprocess, 
                     stopwords, profile_words, args.beta)
    # save computed similarity data to file
    with open(const._SIMILARITY, 'wb') as f:
        pickle.dump(similarity_all, f)

    print('similarity matrices computed and saved to disk')

    adjust_for_time(tid_to_date, similarity_all, args.T) 

    with open(const._SIMILARITY_ADJUSTED, 'wb') as f:
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