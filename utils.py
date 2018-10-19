import jieba
import re
import json
import database
import pickle
import datetime

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

def load_topics(db, attrs, days, min_len, min_replies, min_replies_1, path):
    topics = {}
    attrs = ', '.join(attrs)
    for i in range(10):
        sql = '''SELECT t.TOPICID, ti.body, {}
                 FROM topics_{} as t, topics_info_{} as ti
                 WHERE t.TOPICID = ti.topicID AND 
                 t.POSTDATE BETWEEN 
                 NOW()-INTERVAL {} DAY AND NOW()'''.format(attrs, i, i, days)
        with db.query(sql) as cursor:
            for rec in cursor:
                reply_cnt = int(rec['TOTALREPLYNUM'])
                if reply_cnt < min_replies: 
                    continue 
                if len(rec['body']) < min_len and reply_cnt < min_replies_1:
                    continue 
                if rec['body'] is not None:
                    topics[rec['TOPICID']] = {
                        k:v for k,v in rec.items() if k != 'TOPICID'
                    }

    def convert_datetime(o):
        if isinstance(o, datetime.datetime):
            return o.__str__()

    with open(path, 'w') as f:
        json.dump(topics, f, default=convert_datetime)

    print('过去{}天共计{}条有效主贴'.format(days, len(topics)))
    return list(topics.keys())

def load_replies(db, topic_ids, attrs, path):
    replies = {topic_id:{} for topic_id in topic_ids}
    attrs = ', '.join(attrs)
    for topic_id in topic_ids:
        i = 0
        while i < 10:
            sql = '''SELECT r.TOPICID, r.REPLYID, ri.body, {}
                     FROM replies_{} as r, replies_info_{} as ri
                     WHERE r.REPLYID = ri.replyID AND 
                     r.TOPICID = {}'''.format(attrs, i, i, topic_id)
            with db.query(sql) as cursor:
                results = cursor.fetchall()
            if len(results) == 0:
                i += 1
                continue
            for rec in results:
                if rec['body'] is not None:
                    replies[rec['TOPICID']][rec['REPLYID']] = {
                        k:v for k,v in rec.items() if k not in {'TOPICID', 'REPLYID'} 
                    }
            break

    with open(path, 'w') as f:
        json.dump(replies, f)  

    print('以上主贴共计有{}条跟帖'.format(
           sum([len(replies[topic_id]) for topic_id in replies])))

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