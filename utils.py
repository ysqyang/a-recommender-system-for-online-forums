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
            for (topic_id, ) in cursor:
                mapping[topic_id] = i

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
            for (topic_id, date) in cursor:
                mapping[topic_id] = date

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

def update_tid_to_table_num_mapping(path, db, active_topics):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    def get_new_topic_table(topic_id):
        i = 0
        while i < 10:
            sql = '''SELECT * FROM topics_{} WHERE 
                     TOPICID = {}'''.format(i, topic_id)
            with db.query(sql) as cursor:
                if cursor.fetchone():
                    return i
            i += 1

        return -1

    for topic_id in active_topics:
        if topic_id not in mapping:
            mapping[topic_id] = get_new_topic_table(topic_id)

    with open(path, 'wb') as f:
        pickle.dump(mapping, f)

    return mapping

def update_tid_to_reply_table_num_mapping(path, db, active_topics):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    for topic_id in active_topics:
        if topic_id not in mapping:
            while j < 10:
                sql = '''SELECT * FROM replies_{} WHERE 
                         TOPICID = {}'''.format(j, topic_id)
                with db.query(sql) as cursor:
                    if cursor.fetchone():
                        mapping[topic_id] = j
                        break
                j += 1

    with open(path, 'wb') as f:
        pickle.dump(mapping, f)

    return mapping

def update_tid_to_date_mapping(path, db, active_topics, tid_to_table):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    for topic_id in active_topics:
        if topic_id not in mapping:
            table_num = tid_to_table[topic_id]
            sql = '''SELECT POSTDATE FROM topics_{} WHERE 
                     TOPICID = {}'''.format(table_num, topic_id)
            with db.query(sql) as cursor:
                mapping[topic_id] = cursor.fetchone()[0]

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