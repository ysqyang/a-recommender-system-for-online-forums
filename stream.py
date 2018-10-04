# Class definitions for streaming data from the database

class Corpus_under_topic(object):
    '''
    Corpus object for streaming and preprocessing 
    texts from the topics_info and replies_info tables
    '''
    def __init__(self, database, topic_id, table_num, 
                 preprocess_fn, stopwords):
        self.cursor = database.cursor()
        self.topic_id = topic_id
        self.table_num = table_num
        self.stopwords = stopwords 
        self.preprocess_fn = preprocess_fn
        self.reply_id_to_corpus_index = {}
        
    def __iter__(self):
        # iteration starts with the topic content first
        sql = '''SELECT BODY FROM topics_info_{}
                 WHERE TOPICID = {}'''.format(self.table_num, self.topic_id)
        self.cursor.execute(sql)
        topic_content = ' '.join(self.cursor.fetchone().split())
        yield self.preprocess_fn(topic_content, self.stopwords)
        
        # iterates through replies under this topic id
        for i in range(10):
            sql = '''SELECT REPLYID, BODY FROM replies_info_{}
                     WHERE TOPICID = {}'''.format(i, self.topic_id)
            self.cursor.execute(sql)
            idx = 1
            for (reply_id, content) in self.cursor:
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
        self.cursor = database.cursor()
        self.stopwords = stopwords 
        self.preprocess_fn = preprocess_fn
        self.topic_id_to_corpus_index = {}
        
    def __iter__(self):
        # iterates through all topics
        for i in range(10):
            sql = 'SELECT TOPICID, BODY FROM topics_info_{}'.format(i)
            self.cursor.execute(sql)
            idx = 0
            for (topic_id, content) in self.cursor:
                if content is not None:
                    self.topic_id_to_corpus_index[topic_id] = idx 
                    text = ' '.join(content.split())
                    idx += 1
                    yield self.preprocess_fn(text, self.stopwords)