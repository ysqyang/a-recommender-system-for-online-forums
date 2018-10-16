# Class definitions for streaming data from the database
import pymysql
from pymysql.cursors import Cursor, DictCursor
from gensim import corpora, models
from sklearn import preprocessing
import warnings
import numpy as np
import collections
import math
from scipy import stats

class Database(object):
    def __init__(self, hostname, username, password, dbname, 
                 port, charset):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.dbname = dbname
        self.port = port
        self.charset = charset
        #self.cursorclass = cursorclass
        
    def connect(self):
        self.conn = pymysql.connect(host=self.hostname,
                                    user=self.username,
                                    password=self.password,
                                    db=self.dbname,
                                    port=self.port, 
                                    charset=self.charset)

    def query(self, sql):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
        except:
            print('establishing connection to database...')
            self.connect()
            print('connection established')
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
        self.model = models.TfidfModel

    def __iter__(self):
        # iteration starts with the topic content first
        sql = '''SELECT BODY FROM topics_info_{}
                 WHERE TOPICID = {}'''.format(self.topic_table_num, self.topic_id)
        with self.database.query(sql) as cursor:
            (topic_content, ) = cursor.fetchone()
            topic_content = ' '.join(topic_content.split())
            #print(self.preprocess_fn(topic_content, self.stopwords))
            yield self.preprocess_fn(topic_content, self.stopwords)

        if self.reply_table_num is not None:
            # iterates through replies under this topic id       
            sql = '''SELECT REPLYID, BODY FROM replies_info_{}
                     WHERE TOPICID = {}'''.format(self.reply_table_num, self.topic_id)
            with self.database.query(sql) as cursor:  
                idx = 1
                for (reply_id, content) in cursor:
                    if content is None:
                        continue
                    self.reply_id_to_corpus_index[reply_id] = idx 
                    text = ' '.join(content.split())
                    idx += 1
                    yield self.preprocess_fn(text, self.stopwords)

    def get_dictionary(self):
        self.dictionary = corpora.Dictionary(self)

    def get_scores(self, features, weights, reply_table_num):
        '''
        Computes importance scores for replies
        Args:
        features:        attributes to include in importance evaluation
        weights:         weights associated with attributes in features
        reply_table_num: mapping from topic id to replies table number
        ''' 
        self.scores = {}
        if reply_table_num is None:
            return
            
        s, scaler = sum(weights), preprocessing.MinMaxScaler() 
        norm_weights = [wt/s for wt in weights]  # normalize weights
         
        attrs = ', '.join(['REPLYID']+features)
        sql = '''SELECT {} FROM replies_{}
                 WHERE TOPICID = {}'''.format(attrs, reply_table_num, self.topic_id)

        with self.database.query(sql) as cursor:
            results = cursor.fetchall()
            print('results:', results)
            # normalize features using min-max scaler
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features_norm = scaler.fit_transform(np.array(results)[..., 1:])
                for result, feature_vec in zip(results, features_norm):
                    if result[0] in self.reply_id_to_corpus_index:
                        corpus_index = self.reply_id_to_corpus_index[result[0]]
                        self.scores[corpus_index] = np.dot(feature_vec, norm_weights)

    def get_word_weight(self, alpha=0.7, smartirs='atn'):
        '''
        Computes word importance
        Args:
        bow:      bag-of-words representation of corpus
        alpha:    contribution coefficient for the topic content
        smartirs: tf-idf weighting variants 
        '''
        self.word_weight = collections.defaultdict(float) 
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in self]
        language_model = self.model(corpus_bow, smartirs=smartirs)

        # if there is no replies under this topic, use augmented term frequency
        # as word weight
        if len(corpus_bow) == 1:
            if len(corpus_bow[0]):
                max_freq = max(x[1] for x in corpus_bow[0])
                self.word_weight = {self.dictionary[x[0]]:(1+x[1]/max_freq)/2 
                                    for x in corpus_bow[0]}
            return

        # get the max score under each topic for normalization purposes
        max_score = max(self.scores.values()) + 1e-8  # add 1e-8 to prevent division by zero
        print('max_score for topic {}:'.format(self.topic_id), max_score)

        for i, doc in enumerate(corpus_bow):
            if len(doc) == 0:
                continue
            converted = language_model[doc]
            if len(converted) == 0:
                continue
            max_weight = max([x[1] for x in converted])
            coeff = 1-alpha if i else alpha
            score_norm = self.scores[i]/max_score if i else 1 
            for word_id, weight in converted:
                weight_norm = weight/max_weight
                self.word_weight[self.dictionary[word_id]] += coeff*score_norm*weight_norm

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
        self.corpus_index_to_topic_id = {}
        #self.topic_ids = topic_ids
        
    def __iter__(self):
        # iterates through all topics
        for i in range(1):
            sql = 'SELECT TOPICID, BODY FROM topics_info_{}'.format(i)
            with self.database.query(sql) as cursor:
                idx = 0
                for (topic_id, content) in cursor:
                    if content is None:
                        continue
                    self.topic_id_to_corpus_index[topic_id] = idx 
                    self.corpus_index_to_topic_id[idx] = topic_id
                    text = ' '.join(content.split())
                    idx += 1
                    if idx == 10:
                        return
                    yield self.preprocess_fn(text, self.stopwords)

    def get_dictionary(self):
        self.dictionary = corpora.Dictionary(self)

    def get_word_frequency(self):
        '''
        Computes normalized word frequencies in each document in 
        corpus_all_topics and in the entire corpus_all_topics
        Creates two attributes:
        self.doc_freq:    list of word frequencies in each document
        self.corpus_freq: dictionary of word frequencies in the entire corpus
        '''
        self.doc_freq = []
        self.corpus_freq = collections.defaultdict(int)
        bow_corpus = [self.dictionary.doc2bow(doc) for doc in self]
        num_tokens_corpus = sum(sum(x[1] for x in vec) for vec in bow_corpus)
        # iterate through documents in corpus
        for vec in bow_corpus:
            # total number of tokens (with repetitions) in current doc 
            num_tokens = sum(x[1] for x in vec)
            self.doc_freq.append({x[0]:x[1]/num_tokens for x in vec})
            for (word_id, count) in vec:
                # update word frequency in corpus 
                self.corpus_freq[word_id] += count/num_tokens_corpus

    def get_word_doc_prob(self, coeff):
        '''
        Computes the word probabilities w.r.t. each document in the corpus
        Args:
        coeff: contribution coefficient for in-document word frequency
               in computing word frequency in document
        Creates self.doc_prob: word probabilities w.r.t. each document 
        '''
        self.doc_prob = []
        for vec in self.doc_freq:
            # contribution from word frequency in corpus
            prob = [(1-coeff)*self.corpus_freq[wid] 
                    for wid in self.dictionary.token2id.values()]
            # iterate through all words by id
            for word_id in vec:
                # contribution from word frequency in document
                prob[word_id] += coeff*vec[word_id]
            self.doc_prob.append(prob[:])

    def get_prob_topic_profile(self, profile_words):
        '''
        Computes the conditional probability of observing a word 
        given the apprearance of profile_words
        Args:
        profile_words: list of words that represent a discussion thread
                       (i.e., topic with all replies)
        Returns:
        Conditional probability of observing a word given the apprearance of profile_words
        '''
        profile_word_ids = []
        for word in profile_words:
            if word in self.dictionary.token2id:
                profile_word_ids.append(self.dictionary.token2id[word])

        print(profile_word_ids)
        prob = [0]*len(self.dictionary.token2id)
        # compute the join probability of observing each dictionary
        # word together with profile words 
        for word_id in self.dictionary.token2id.values():        
            # iterate through each document in corpus
            for vec in self.doc_prob:
                # compute the joint probability for each doc
                # convert to natural log to avoid numerical issues
                log_prob = 0 if word_id in profile_word_ids else math.log(vec[word_id])
                for profile_word_id in profile_word_ids:
                    log_prob += math.log(vec[profile_word_id])
                # assuming uniform prior distribution over all docs in corpus,
                # the joint probability is the sum of joint probabilities over
                # all docs
                prob[word_id] += math.exp(log_prob)

        # normalize the probabilities
        s = sum(prob)
        for i in range(len(prob)):
            prob[i] /= s
        
        return prob

    def get_similarity(self, prob_topic_profile):
        '''
        Computes the similarity scores between a topic profile and 
        the documents in the corpus
        Args:
        prob_topic_profile: word probabilities given a topic profile
        Returns:
        Similarity scores between a topic profile and the documents 
        in the corpus
        '''
        similarities = {}
        for i, vec in enumerate(self.doc_prob):
            topic_id = self.corpus_index_to_topic_id[i]
            similarities[topic_id] = stats.entropy(pk=prob_topic_profile, qk=vec)

        return similarities