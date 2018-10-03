import pickle
import jieba

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

def load_topic_id_to_table_num_dict(path):
    '''
    Loads the mapping from topic id to table number from disk:
    Args:
    path: path of file containing the mapping
    Returns:
    A dictionary containing topic id to table number mapping
    '''
    try:
        with open(path, 'rb') as f:
            mapping = pickle.load(f)
        return mapping
    except:
        print('File does not exist. Run utilities.py first to create it')

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