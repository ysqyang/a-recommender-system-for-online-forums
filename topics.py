# Class definitions for streaming data from file
from gensim import corpora, models
from sklearn import preprocessing
import warnings
import numpy as np
import collections
import math
import json
import constants as const
from scipy import stats
from datetime import datetime

class Topic(object):
    '''
    Corpus object for streaming and preprocessing 
    texts from the topics_info and replies_info tables
    '''
    def __init__(self, topic_id):
        self.topic_file = const._TOPIC_FILE
        self.reply_file = const._REPLY_FILE
        self.topic_id = topic_id

    def make_corpus_with_scores(self, filter_fn, preprocess_fn, stopwords,
                                features, weights):
        '''
        Creates a corpus for this topic and computes importance scores
        for replies
        Args:
        preprocess_fn: function to preprocess a document
        stopwords:     set of stopwords 
        features:      attributes to include in importance evaluation
        weights:       weights associated with attributes in features
        '''
        self.corpus, self.scores, feature_matrix = [], [], []
        # iteration starts with the topic content first
        with open(self.topic_file, 'r') as f:
            data = json.load(f)
        content = data[str(self.topic_id)]['body']
        if not filter_fn(content):
            self.valid = False
            return
        self.valid = True
        content = ' '.join(content.split())
        word_list = preprocess_fn(content, stopwords)
        #print(self.preprocess_fn(topic_content, self.stopwords))
        if len(word_list) > 0:
            self.corpus.append(word_list)

        with open(self.reply_file, 'r') as f:
            # iterates through replies under this topic id       
            data = json.load(f)
        
        replies = data[str(self.topic_id)]

        for reply_id, rec in replies.items():
            content = ' '.join(rec['body'].split())
            word_list = preprocess_fn(content, stopwords)
            if len(word_list) > 0:
                feature_vec = [rec[feature] for feature in features]
                feature_matrix.append(feature_vec)
                self.corpus.append(word_list)
            
        if len(feature_matrix) == 0:
            return
        s, scaler = sum(weights), preprocessing.MinMaxScaler() 
        if s == 0:
            raise ValueError('weights cannot be all zeros')
        norm_weights = [wt/s for wt in weights]  # normalize weights
         
        # normalize features using min-max scaler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_matrix = scaler.fit_transform(feature_matrix)
            for feature_vec in self.feature_matrix:
                self.scores.append(np.dot(feature_vec, norm_weights))

        #normalize scores
        max_score = max(self.scores) + 1e-8 # add 1e-8 to prevent division by zero
        for i in range(len(self.scores)):
            self.scores[i] /= max_score

    def get_dictionary(self):
        self.dictionary = corpora.Dictionary(self.corpus)

    def get_word_weight(self, alpha=0.7, smartirs='atn'):
        '''
        Computes word importance
        Args:
        bow:      bag-of-words representation of corpus
        alpha:    contribution coefficient for the topic content
        smartirs: tf-idf weighting variants 
        '''
        self.word_weight = collections.defaultdict(float) 
        if len(self.corpus) == 0:
            return

        corpus_bow = [self.dictionary.doc2bow(doc) for doc in self.corpus]
        # if there is no replies under this topic, use augmented term frequency
        # as word weight
        if len(corpus_bow) == 1:
            if len(corpus_bow[0]) > 0:
                max_freq = max(x[1] for x in corpus_bow[0])
                self.word_weight = {self.dictionary[x[0]]:(1+x[1]/max_freq)/2 
                                    for x in corpus_bow[0]}
            return

        model = models.TfidfModel(corpus_bow, smartirs=smartirs)
        
        tfidf = corpus_bow[0]
        max_weight = max([x[1] for x in tfidf])
        for word_id, weight in tfidf:
            self.word_weight[self.dictionary[word_id]] += alpha*weight/max_weight

        for doc, score in zip(corpus_bow[1:], self.scores):
            tfidf = model[doc]
            if len(tfidf) == 0:
                continue
            max_weight = max([x[1] for x in tfidf])
            for wid, weight in tfidf:
                weight /= max_weight
                if score > 1e-8:
                    self.word_weight[self.dictionary[wid]] += (1-alpha)*score*weight

class Topic_collection(object):
    '''
    Corpus object for streaming and preprocessing 
    texts from topics_info tables
    '''
    def __init__(self, topic_ids):
        self.topic_file = const._TOPIC_FILE
        self.topic_ids = topic_ids
        self.valid_topics = []
        
    def make_corpus(self, filter_fn, preprocess_fn, stopwords):
        # iterates through all topics
        self.corpus, self.dates = [], []
        with open(self.topic_file, 'r') as f:
            data = json.load(f)
            for topic_id in self.topic_ids:
                topic = data[str(topic_id)]
                if not filter_fn(topic['body']):
                    continue
                content = ' '.join(topic['body'].split())
                word_list = preprocess_fn(content, stopwords)
                word_set = set(word_list)
                if len(word_set) > 0 and len(word_list)/len(word_set) > const._VALID_RATIO:
                    self.corpus.append(word_list)
                    self.dates.append(topic['POSTDATE'])
                    self.valid_topics.append(topic_id)

    def get_dictionary(self):
        self.dictionary = corpora.Dictionary(self.corpus)

    def get_distributions(self, coeff):
        '''
        Computes the word distribution for each document in the collection 
        from in-document and in-corpus frequencies
        Args:
        coeff: contribution coefficient for in-document word frequency
               in computing word distribution
        '''
        n_words = len(self.dictionary.token2id)
        self.distributions = []
        corpus_freq = [0]*n_words
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in self.corpus]
        num_tokens_corpus = sum(sum(x[1] for x in vec) for vec in corpus_bow)

        for i, vec in enumerate(corpus_bow):
            # ignore documents that have less than _MIN_WORDS words 
            if len(vec) == 0:
                self.distributions.append([])
            else:
                # total number of tokens (with repetitions) in current doc
                num_tokens = sum(x[1] for x in vec)
                dst = [0]*n_words
                for (word_id, count) in vec:
                    dst[word_id] = coeff*count/num_tokens
                    # update word frequency in corpus
                    corpus_freq[word_id] += count/num_tokens_corpus
                self.distributions.append(dst)
                '''
                dst = [(self.dictionary[i], val) for i, val in enumerate(dst)]
                dst.sort(key=lambda x:x[1], reverse=True)
                print([(word, val) for word, val in dst if val > 0])
                '''
        for i, dst in enumerate(self.distributions):
            #print(self.topic_ids[i], self.corpus[i])
            if len(dst) > 0:
                for word_id in range(n_words):
                    # add contribution from in-corpus frequency
                    dst[word_id] += (1-coeff)*corpus_freq[word_id]
                '''
                dst1 = [(self.dictionary[i], val) for i, val in enumerate(dst)]
                dst1.sort(key=lambda x:x[1], reverse=True)
                print([word for word, val in dst1[:20]])
                '''

    def get_distribution_given_profile(self, profile_words):
        '''
        Computes the word distribution given the apprearance of profile_words
        Args:
        profile_words: list of words that represent a discussion thread
                       (i.e., topic with all replies)
        Returns:
        Word distribution given the apprearance of profile_words
        '''
        profile_wids = []
        for word in profile_words:
            if word in self.dictionary.token2id:
                profile_wids.append(self.dictionary.token2id[word])

        #print(profile_word_ids)
        distribution = [0]*len(self.dictionary.token2id)
        # compute the joint probability of observing each dictionary
        # word together with profile words 

        # convert to natural log to avoid numerical issues
        log_probs = [sum(math.log(v[wid]) for wid in profile_wids)
                     if len(v) > 0 else -float('inf') for v in self.distributions]
        # assuming uniform prior distribution over all docs in corpus,
        # the joint probability is the sum of joint probabilities over
        # all docs
        for wid in self.dictionary.token2id.values():        
            for v, log_prob in zip(self.distributions, log_probs):
                if len(v) > 0:
                    log_prob += math.log(v[wid])
                distribution[wid] += math.exp(log_prob)

        # normalize the probabilities
        s = sum(distribution)
        for i in range(len(distribution)):
            distribution[i] /= s
        
        return distribution

    def get_similarity(self, distribution, T):
        '''
        Computes the similarity scores between a topic profile and 
        the documents in the corpus with time adjustments
        Args:
        distribution: word probability distribution given a topic profile
        T:            time attenuation factor
        Returns:
        Similarity scores between a topic profile and the documents 
        in the corpus
        '''
        sim, now = {}, datetime.now() 
        for date, vec, tid in zip(self.dates, self.distributions, self.valid_topics):
            if len(vec) > 0:     
                date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                day_diff = (now - date).days
                sim[tid] = stats.entropy(pk=distribution, qk=vec)*math.exp(day_diff/T)
                '''
                if sim[self.topic_ids[idx]] < 0.1:
                    print(self.topic_ids[idx], self.corpus[idx])
                    dst = [(self.dictionary[i], val) for i, val in enumerate(vec)]
                    dst.sort(key=lambda x:x[1], reverse=True)
                    print([(word, val) for word, val in dst[:20]])
                    print(self.topic_ids[idx], sim[self.topic_ids[idx]])
                '''
        return sim