import pickle
import jieba

def preprocess(text, stopwords):
    '''
    Tokenize a Chinese document to a list of words 
    Args:
    text:      text to be tokenized
    stopwords: set of stopwords
    '''  
    return [word for word in jieba.lcut(text) if word not in stopwords]  

def topic_id_to_table_number(db, path):
    '''
    Builds a mapping from topic id to database table number
    (topics_info_?) and saves to disk
    Args:
    db:   pymysql connection object
    path: path to save the dictionary 
    '''
    mapping = {}
    cursor = db.cursor()

    for i in range(10):
        sql = 'SELECT TOPICID FROM topics_info_{}'.format(i)
        cursor.execute(sql)
        for topic_id in cursor:
            mapping[topic_id] = i

    with open(path, 'wb') as f:
        pickle.dump(mapping, f)