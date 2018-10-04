import pickle
import jieba
import pymysql

def load_stopwords(stopwords_path):
    '''
    Creates a set of stopwords from a file specified by
    soptowrds_path
    Args:
    stopwords_path: path of stopwords file
    Returns:
    A set containing all stopwords
    '''
    stopwords = set()
    # load stopwords dictionary to create stopword set
    with open(stopwords_path, 'r') as f:
        n = 1
        while True:
            stopword = f.readline().strip('\n')
            if stopword == '':
                break
            stopwords.add(stopword)
            n += 1

    return stopwords|{'\n', ' '}

def load_topic_id_to_table_num(db, path):
    '''
    Loads the mapping from topic id to table number from disk:
    Args:
    db:   database connection
    path: path of file containing the mapping
    Returns:
    A dictionary containing topic id to table number mapping
    '''
    try:
        with open(path, 'rb') as f:
            mapping = pickle.load(f)
    except:
        mapping = {}
        cursor = db.cursor()
        for i in range(10):
            sql = 'SELECT TOPICID FROM topics_info_{}'.format(i)
            cursor.execute(sql)
            for topic_id in cursor:
                mapping[topic_id] = i

        with open(path, 'wb') as f:
            pickle.dump(mapping, f)

    return mapping

def connect_to_database(db_info):
    '''
    Connect to a database specified by db_info
    Args:
    db_info:  (host, user, password, database, port)
    Returns:
    A pymysql database connection object
    '''
    return pymysql.connect(*db_info)

def preprocess(text, stopwords):
    '''
    Tokenize a Chinese document to a list of words 
    Args:
    text:      text to be tokenized
    stopwords: set of stopwords
    '''  
    return [word for word in jieba.lcut(text) if word not in stopwords]   