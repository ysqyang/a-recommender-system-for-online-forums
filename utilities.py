import pickle
import jieba
import pymysql
import stream

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
            stopword = f.readline()
            if stopword == '':
                break
            stopwords.add(stopword.strip('\n'))
            n += 1

    return stopwords|{'\n', ' '}

def get_database(db_info):
    '''
    Connect to a database specified by db_info
    Args:
    db_info:  (host, user, password, database, port, charset)
    Returns:
    A pymysql database connection object
    '''
    return stream.Database(*db_info)

def load_topic_id_to_table_num(db, path):
    '''
    Loads the mapping from topic id to table number from disk:
    Args:
    db:   database 
    path: path of file containing the mapping
    Returns:
    A dictionary containing topic id to table number mapping
    '''
    try:
        with open(path, 'rb') as f:
            mapping = pickle.load(f)
    except:
        mapping = {}
        for i in range(10):
            sql = 'SELECT TOPICID FROM topics_info_{}'.format(i)
            with db.query(sql) as cursor:
                for (topic_id,) in cursor:
                    mapping[topic_id] = i

        with open(path, 'wb') as f:
            pickle.dump(mapping, f)

    return mapping

def load_topic_id_to_date(db, path):
    '''
    Loads the mapping from topic id to posting date from disk:
    Args:
    db:   database
    path: path of file containing the mapping
    Returns:
    A dictionary containing topic id to posting date mapping
    '''
    try:
        with open(path, 'rb') as f:
            mapping = pickle.load(f)
    except:
        mapping = {}
        for i in range(10):
            sql = 'SELECT TOPICID, POSTDATE FROM topics_{}'.format(i)
            with db.query(sql) as cursor:    
                for topic_id, date in cursor:
                    mapping[topic_id] = date

        with open(path, 'wb') as f:
            pickle.dump(mapping, f)

    return mapping

def load_topic_id_to_reply_table(db, topic_ids, path):
    '''
    Loads the mapping from topic id to reply table number from disk:
    Args:
    db:        database 
    topic_ids: list of al topic_ids
    path:      path of file containing the mapping
    Returns:
    A dictionary containing topic id to reply table number mapping
    '''
    try:
        with open(path, 'rb') as f:
            mapping = pickle.load(f)
    except:
        print('Creating topic-id-to-reply-table mapping...')
        mapping, percentage = {}, .05
        for i, tid in enumerate(topic_ids):
            j = 0
            while j < 10:
                sql = 'SELECT * FROM replies_{} WHERE TOPICID = {}'.format(j, tid)
                with db.query(sql) as cursor:
                    if cursor.fetchone() is not None:
                        mapping[tid] = j
                        #print(tid, i)
                        break
                j += 1

            print(len(mapping))

        with open(path, 'wb') as f:
            pickle.dump(mapping, f)

    return mapping

def preprocess(text, stopwords):
    '''
    Tokenize a Chinese document to a list of words 
    Args:
    text:      text to be tokenized
    stopwords: set of stopwords
    '''  
    return [word for word in jieba.lcut(text) if word not in stopwords]   