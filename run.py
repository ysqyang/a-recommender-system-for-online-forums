import stream
import utilities
import topic_profiling
import similarity
import argparse
import time
from gensim import models

_STOPWORDS = './stopwords.txt'
_DB_INFO = ('192.168.1.102','tgbweb','tgb123321','taoguba',3307)
_TOPIC_ID_TO_TABLE_NUM = './topic_id_to_table_num'
_TOPIC_ID_TO_DATE = './topic_id_to_date'
_IMPORTANCE_FEATURES = [] 
_WEIGHTS = []
_SAVE_PATH_1 = './word_importance'
_SAVE_PATH_2 = './similarity'
_MODEL = models.TfidfModel

def main(args):
    stopwords = utilities.load_stopwords(_STOPWORDS)
    db = utilities.connect_to_database(DB_INFO)
    tid_to_table = utilities.load_topic_id_to_table_num(db, _TOPIC_ID_TO_TABLE_NUM)
    tid_to_date = utilities.load_topic_id_to_date(db, _TOPIC_ID_TO_DATE)
    word_weight = topic_profiling.get_word_weights_all(
                                db, tid_to_table, _IMPORTANCE_FEATURES, _WEIGHTS, 
                                utilities.preprocess, stopwords, args.alpha, 
                                args.smartirs)

    # save the word weights dictionary to file
    with open(_SAVE_PATH_1, 'w') as f:
        pickle.dump(word_weight, f)

    # get k most representative words for each topic
    profile_words = {tid:topic_profiling.get_top_k_words(weight, args.k)
                     for tid, weight in word_weight.items()}

    similarity_all = similarity.get_similarity_all(db, utilities.preprocess, stopwords, 
                                    profile_words, args.beta)

    with open(_SAVE_PATH_2, 'w') as f:
        pickle.dump(similarity_all, f)

    adjust_for_time_decay(tid_to_index, tid_to_date, similarity_all, args.T) 




if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('alpha', type=float, default=0.7, 
                        help='''contribution coefficient for topic content 
                                in computing word weights''')
    parser.add_argument('k', type=int, default=60, 
                        help='number of words to represent a discussion thread')
    parser.add_argument('beta', type=float, default=0.5,
                        help='''contribution coefficient for in-document frequency
                                in computing word probabilities''')
    parser.add_argument('T', type=float, default=365, help='time attenuation factor')
    parser.add_argument('smartirs', type=str, default='atn', help='type of tf-idf variants')

    args = parser.parse_args()
    main(args)