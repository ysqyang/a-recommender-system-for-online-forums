import pymysql
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
    db = pymysql.connect(*db_info)
    tid_to_table = utilities.load_topic_id_to_table_num_dict(db, _TOPIC_ID_TO_TABLE_NUM)
    word_weights = {}

    # create a Corpus_under_topic object for each topic
    for topic_id in tid_to_table:
        corpus = stream.Corpus_under_topic(db, topic_id, tid_to_table, 
                                stopwords, utilities.preprocess)
        dictionary = corpora.Dictionary(corpus)
        scores = topic_profiling.compute_scores(db, topic_id, _IMPORTANCE_FEATURES, 
                                weights, corpus.reply_id_to_corpus_index)        
        word_weights[topic_id] = topic_profiling.word_weights(corpus, dictionary, 
                                corpus.topic_id, model, normalize, scores)

    with open(_SAVE_PATH, 'w') as f:
        pickle.dump(word_weights, f)

if __name__ == '__main__':
    main()