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
    db_info:  (host, user, password, database, port, charset, cursorclass)
    Returns:
    A pymysql database connection object
    '''
    return stream.Database(*db_info)

def create_topic_id_to_table_num(db, path):
    mapping = {}
    for i in range(10):
        sql = 'SELECT TOPICID FROM topics_info_{}'.format(i)
        with db.query(sql) as cursor: 
            for record in cursor:
                mapping[record['topicid']] = i

    with open(path, 'wb') as f:
        pickle.dump(mapping, f)

    return mapping

def create_topic_id_to_reply_table(db, topic_ids, path):
    print('Creating mapping from topic id to reply table number...')
    mapping = {}
    for tid in topic_ids:
        j = 0
        while j < 10:
            sql = 'SELECT * FROM replies_{} WHERE TOPICID = {}'.format(j, tid)
            with db.query(sql) as cursor:
                if cursor.fetchone():
                    mapping[tid] = j
                    break
            j += 1

    with open(path, 'wb') as f:
        pickle.dump(mapping, f)

    return mapping

def create_topic_id_to_date(db, path):
    mapping = {}
    for i in range(10):
        sql = 'SELECT TOPICID, POSTDATE FROM topics_{}'.format(i)
        with db.query(sql) as cursor:    
            for record in cursor:
                mapping[rec['TOPICID']] = rec['POSTDATE']

    with open(path, 'wb') as f:
        pickle.dump(mapping, f)

    return mapping

def load_mapping(path):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)
    return mapping

def update_tid_to_table_num_mapping(new_topic_records, path):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    print('number of entries before update: ', len(mapping))

    for rec in new_topic_records:
        mapping[rec['TOPICID']] = rec['USERID']%10

    print('number of entries after update: ', len(mapping))
    
    with open(path, 'wb') as f:
        pickle.dump(mapping, f)

    return mapping

def update_tid_to_reply_table_num_mapping(db, new_topic_records, path):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    print('number of entries before update: ', len(mapping))

    for rec in new_topic_records:
        tid, j = rec['TOPICID'], 0
        while j < 10:
            sql = 'SELECT * FROM replies_{} WHERE TOPICID = {}'.format(j, tid)
            with db.query(sql) as cursor:
                if cursor.fetchone():
                    mapping[tid] = j
                    break
            j += 1

    print('number of entries after update: ', len(mapping))

    with open(path, 'wb') as f:
        pickle.dump(mapping, f)

    return mapping

def update_tid_to_date_mapping(new_topic_records, path):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    print('number of entries before update: ', len(mapping))

    for rec in new_topic_records:
        mapping[rec['TOPICID']] = rec['POSTDATE'] 

    print('number of entries after update: ', len(mapping))

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