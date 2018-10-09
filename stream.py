# Class definitions for streaming data from the database
import pymysql

class Database(object):
    def __init__(self, hostname, username, password, dbname, 
                 port):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.dbname = dbname
        self.port = port
        
    def connect(self):
        self.conn = pymysql.connect(host=self.hostname,
                                    user=self.username,
                                    password=self.password,
                                    db=self.dbname,
                                    port=self.port)

    def query(self, sql):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
        except:
            self.connect()
            cursor = self.conn.cursor()
            cursor.execute(sql)
        return cursor

class Corpus_under_topic(object):
    '''
    Corpus object for streaming and preprocessing 
    texts from the topics_info and replies_info tables
    '''
    def __init__(self, database, topic_id, topic_table_num, 
                 reply_table_num, preprocess_fn, stopwords):
        self.database = database
        self.topic_id = topic_id
        self.topic_table_num = topic_table_num
        self.reply_table_num = reply_table_num
        self.stopwords = stopwords 
        self.preprocess_fn = preprocess_fn
        self.reply_id_to_corpus_index = {}

    def __iter__(self):
        # iteration starts with the topic content first
        sql = '''SELECT BODY FROM topics_info_{}
                 WHERE TOPICID = {}'''.format(self.topic_table_num, self.topic_id)
        with self.database.query(sql) as cursor:
            (topic_content, ) = cursor.fetchone()
            topic_content = ' '.join(topic_content.split())
            yield self.preprocess_fn(topic_content, self.stopwords)

        # iterates through replies under this topic id       
        sql = '''SELECT REPLYID, BODY FROM replies_info_{}
                 WHERE TOPICID = {}'''.format(self.reply_table_num, self.topic_id)
        with self.database.query(sql) as cursor:  
            idx = 1
            for (reply_id, content) in cursor:
                if content is not None:
                    self.reply_id_to_corpus_index[reply_id] = idx 
                    text = ' '.join(content.split())
                    idx += 1
                    yield self.preprocess_fn(text, self.stopwords)

class Corpus_all_topics(object):
    '''
    Corpus object for streaming and preprocessing 
    texts from topics_info tables
    '''
    def __init__(self, database, preprocess_fn, stopwords):
        self.database = database
        self.stopwords = stopwords 
        self.preprocess_fn = preprocess_fn
        self.topic_id_to_corpus_index = {}
        
    def __iter__(self):
        # iterates through all topics
        for i in range(10):
            sql = 'SELECT TOPICID, BODY FROM topics_info_{}'.format(i)
            with self.database.query(sql) as cursor:
                idx = 0
                for (topic_id, content) in cursor:
                    if content is None:
                        continue
                    self.topic_id_to_corpus_index[topic_id] = idx 
                    text = ' '.join(content.split())
                    idx += 1
                    yield self.preprocess_fn(text, self.stopwords)

