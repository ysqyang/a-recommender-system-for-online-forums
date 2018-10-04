import topic_profiling
import utilities
import stream
from gensim import corpora

_STOPWORDS = './stopwords.txt'
_DB_INFO = ('192.168.1.102','tgbweb','tgb123321','taoguba',3307)
_TOPIC_ID_TO_TABLE_NUM = './topic_id_to_table_num'
_IMPORTANCE_FEATURES = [] 
_WEIGHTS = []
_SAVE_PATH = './word_importance'

def main():
    stopwords = utilities.load_stopwords(_STOPWORDS)
    db = utilities.connect_to_database(DB_INFO)
    tid_to_table = utilities.load_topic_id_to_table_num(db, _TOPIC_ID_TO_TABLE_NUM)

    word_weights = topic_profiling.get_word_weights_all_topics(
                                db, tid_to_table, _IMPORTANCE_FEATURES, _WEIGHTS, 
                                utilities.preprocess, stopwords, normalize)

    # save the word weights dictionary to file
    with open(_SAVE_PATH, 'w') as f:
        pickle.dump(word_weights, f)

    # 




if __name__ == '__main__':
    main()