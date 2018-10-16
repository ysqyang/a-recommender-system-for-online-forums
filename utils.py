import pickle
import jieba
import pymysql
import stream
import re

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

    print('Created topic-id-to-table-number mapping with {} entries'.format(len(mapping)))
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

    print('Created topic-id-to-reply-table-number mapping with {} entries'.format(len(mapping)))
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

    print('Created topic-id-to-date mapping with {} entries'.format(len(mapping)))
    return mapping

def load_mapping(path):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)
    return mapping

def get_new_topics(db, existing):
    new_topic_records = []

    for i in range(10):
        sql = 'SELECT TOPICID FROM topics_{}'.format(i)
        with db.query(sql) as cursor:
            for (topic_id,) in cursor:
                if topic_id not in existing:
                    new_topic_records.append(topic_id)

    print('Found {} new topics'.format(len(new_topic_records)))
    return new_topic_records

def update_tid_to_table_num_mapping(path, db, new_topics):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    print('entries before update: ', len(mapping))
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

    for topic_id in new_topics:
        if topic_id not in mapping:
            mapping[topic_id] = get_new_topic_table(topic_id)

    print('entries after update: ', len(mapping))
    with open(path, 'wb') as f:
        pickle.dump(mapping, f)

    return mapping

def update_tid_to_reply_table_num_mapping(path, db, new_topics):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    print('entries before update: ', len(mapping))
    for topic_id in new_topics:
        if topic_id not in mapping:
            j = 0
            while j < 10:
                sql = '''SELECT * FROM replies_{} WHERE 
                         TOPICID = {}'''.format(j, topic_id)
                with db.query(sql) as cursor:
                    if cursor.fetchone():
                        mapping[topic_id] = j
                        break
                j += 1

    print('entries after update: ', len(mapping))
    with open(path, 'wb') as f:
        pickle.dump(mapping, f)

    return mapping

def update_tid_to_date_mapping(path, db, new_topics, tid_to_table):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    print('entries before update: ', len(mapping))

    for topic_id in new_topics:
        if topic_id not in mapping:
            table_num = tid_to_table[topic_id]
            sql = '''SELECT POSTDATE FROM topics_{} WHERE 
                     TOPICID = {}'''.format(table_num, topic_id)
            with db.query(sql) as cursor:
                mapping[topic_id] = cursor.fetchone()[0]

    print('entries after update: ', len(mapping))
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
    num = r'\d+\.*\d+'
    word_list = []
    words = jieba.cut(text, cut_all=False)
    for word in words:
        if word in stopwords or re.match(num, word):
            continue
        word_list.append(word) 
    return word_list   