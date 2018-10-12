import utils
import topic_profiling as tp
import similarity
import argparse
from gensim import models
from pymysql.cursors import Cursor

_STOPWORDS = './stopwords.txt'
_DB_INFO = ('192.168.1.102','tgbweb','tgb123321','taoguba', 3307, 'utf8mb4', Cursor)
_TOPIC_ID_TO_TABLE_NUM = './topic_id_to_table_num'
_TOPIC_ID_TO_DATE = './topic_id_to_date'
_TOPIC_ID_TO_REPLY_TABLE_NUM = './topic_id_to_reply_table_num'
_IMPORTANCE_FEATURES = ['USEFULNUM', 'GOLDUSEFULNUM', 'TOTALPCPOINT'] 
_WEIGHTS = [1, 1, 1]
_SAVE_PATH_WORD_WEIGHT = './word_importance'
_SAVE_PATH_SIMILARITY = './similarity'
_SAVE_PATH_SIMILARITY_ADJUSTED = './similarity_adjusted'

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
    stopwords = utils.load_stopwords(_STOPWORDS)
    db = utils.get_database(_DB_INFO)
    tid_to_table = utils.load_mapping(_TOPIC_ID_TO_TABLE_NUM)
    tid_to_reply_table = utils.load_mapping(_TOPIC_ID_TO_REPLY_TABLE_NUM)
    tid_to_date = utils.load_mapping(_TOPIC_ID_TO_DATE)

    word_weight = tp.get_word_weight_all(
                        db, tid_to_table, tid_to_reply_table, _IMPORTANCE_FEATURES, 
                        _WEIGHTS, utils.preprocess, stopwords, args.alpha, 
                        args.smartirs)
    
    # save computed word weight data to file
    with open(_SAVE_PATH_WORD_WEIGHT, 'wb') as f:
        pickle.dump(word_weight, f)

    print('word weights computed and saved to disk')

    # get k most representative words for each topic
    profile_words = {tid:tp.get_top_k_words(weight, args.k)
                     for tid, weight in word_weight.items()}

    similarity_all = similarity.get_similarity_all(db, utils.preprocess, 
                     stopwords, profile_words, args.beta)
    # save computed similarity data to file
    with open(_SAVE_PATH_SIMILARITY, 'wb') as f:
        pickle.dump(similarity_all, f)

    print('similarity matrices computed and saved to disk')

    adjust_for_time(tid_to_date, similarity_all, args.T) 

    with open(_SAVE_PATH_SIMILARITY_ADJUSTED, 'wb') as f:
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