import pickle
import jieba
import pymysql
import stream

def load_stopwords(stopwords_path):
    stopwords = set()
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

def get_new_topics(db, existing):
    new_topic_records = []

    for i in range(10):
        sql = 'SELECT * FROM topics_{}'.format(i)
        with db.query(sql) as cursor:
            for rec in cursor:
                if rec['TOPICID'] not in existing:
                    new_topic_records.append(rec)

    print('Found {} new topics'.format(len(new_topic_records)))
    return new_topic_records

def update_tid_to_table_num_mapping(new_topics, path):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    for rec in new_topics:
        mapping[rec['TOPICID']] = rec['USERID']%10
    
    with open(path, 'wb') as f:
        pickle.dump(mapping, f)

    return mapping

def update_tid_to_reply_table_num_mapping(db, new_topics, path):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    for rec in new_topics:
        tid, j = rec['TOPICID'], 0
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

def update_tid_to_date_mapping(new_topics, path):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    for rec in new_topics:
        mapping[rec['TOPICID']] = rec['POSTDATE'] 

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