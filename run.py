import pymysql
import topic_profiling
import utilities
import stream

_STOPWORDS = './stopwords.txt'
_DB_INFO = ('192.168.1.102','tgbweb','tgb123321','taoguba',3307)
_TOPIC_ID_TO_TABLE_NUM = './topic_id_to_table_num'
_IMPORTANCE_FEATURES = [] 
_WEIGHTS = []
_SAVE_PATH = './word_importance'

def establish_database_connection(db_info):
    '''
    Establishes a connection to the database specified by db_info
    Args:
    db_info: a tuple of (host, user, password, database, port)
    Returns:
    A database connection object
    '''
    return pymysql.connect(*db_info)

def load_topic_id_to_table_num_dict(path):
	'''
	Loads the mapping from topic id to table number from disk:
	Args:
	path: path of file containing the mapping
	Returns:
	A dictionary containing topic id to table number mapping
	'''
	try:
		with open(path, 'rb') as f:
        	mapping = pickle.load(f)
        return mapping
    except:
    	print('File does not exist. Run utilities.py first to create it')

def main():
	stopwords = utilities.load_stopwords(_STOPWORDS)
    db = establish_database_connection(DB_INFO)
    tid_to_table = load_topic_id_to_table_num_dict(db, _TOPIC_ID_TO_TABLE_NUM)
    word_weights = {}

    # create a Corpus_under_topic object for each topic
    for topic_id in tid_to_table:
        corpus = stream.Corpus_under_topic(db, topic_id, tid_to_table, 
        	                    stopwords, utilities.preprocess)
        dictionary = topic_profiling.build_dictionary(corpus)
        scores = topic_profiling.compute_scores(db, topic_id, _IMPORTANCE_FEATURES, 
                                weights, corpus.reply_id_to_corpus_index)        
        word_weights[topic_id] = topic_profiling.word_weights(corpus, dictionary, 
        	                    corpus.topic_id, model, normalize, scores)

    with open(_SAVE_PATH, 'w') as f:
        pickle.dump(word_weights, f)

if __name__ == '__main__':
	main()