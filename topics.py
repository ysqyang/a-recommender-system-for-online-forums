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

    def make_corpus_with_scores(self, preprocess_fn, stopwords,
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
        content = ' '.join(data[self.topic_id]['body'].split())
        word_list = prepocess_fn(content, stopwords)
        #print(self.preprocess_fn(topic_content, self.stopwords))
        if len(word_list) > 0:
            self.corpus.append(word_list)

        with open(self.reply_file, 'r') as f:
            # iterates through replies under this topic id       
            data = json.load(f)
        
        for reply_id, rec in data[self.topic_id].items():
            content = ' '.join(rec['body'].split())
            word_list = preprocess_fn(content, stopwords)
            if len(word_list) > 0:
                feature_vec = [rec[feature] for feature in features]
                feature_matrix.append(feature_vec)
                self.corpus.append(word_list)
            
        s, scaler = sum(weights), preprocessing.MinMaxScaler() 
        if s == 0:
            print('weights cannot be all zeros')
            raise 
        norm_weights = [wt/s for wt in weights]  # normalize weights
         
        # normalize features using min-max scaler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_matrix = scaler.fit_transform(feature_matrix)
            for feature_vec in self.feature_matrix:
                self.scores.append(np.dot(feature_vec, norm_weights))

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
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in self.corpus]
        language_model = models.TfidfModel(corpus_bow, smartirs=smartirs)
        '''
        # if there is no replies under this topic, use augmented term frequency
        # as word weight
        if len(corpus_bow) == 1:
            if len(corpus_bow[0]):
                max_freq = max(x[1] for x in corpus_bow[0])
                self.word_weight = {self.dictionary[x[0]]:(1+x[1]/max_freq)/2 
                                    for x in corpus_bow[0]}
            return
        '''
        # get the max score under each topic for normalization purposes
        max_score = max(self.scores) + 1e-8  # add 1e-8 to prevent division by zero
        #print('max_score for topic {}:'.format(self.topic_id), max_score)

        for i, doc in enumerate(corpus_bow):
            converted = language_model[doc]
            if len(converted) == 0:
                continue
            max_weight = max([x[1] for x in converted])
            coeff = 1-alpha if i > 0 else alpha
            score_norm = self.scores[i]/max_score if i > 0 else 1 
            for word_id, weight in converted:
                weight_norm = weight/max_weight
                self.word_weight[self.dictionary[word_id]] += coeff*score_norm*weight_norm

class Topic_collection(object):
    '''
    Corpus object for streaming and preprocessing 
    texts from topics_info tables
    '''
    def __init__(self, topic_ids, preprocess_fn, stopwords):
        self.topic_file = const._TOPIC_FILE
        self.topic_ids = topic_ids
        
    def make_corpus(self, preprocess_fn, stopwords):
        # iterates through all topics
        self.corpus, self.dates = [], []
        with open(self.topic_file, 'r') as f:
            data = json.load(f)
            for topic_id in self.topic_ids:
                content = ' '.join(data[topic_id]['body'].split())
                word_list = preprocess_fn(content, stopwords)
                if len(word_list) > 0:
                    self.corpus.append(word_list)
                    self.dates.append(data[topic_id]['POSTDATE'])

    def get_dictionary(self):
        self.dictionary = corpora.Dictionary(self.corpus)

    def get_distributions(self, coeff):
        '''
        Computes the word distribution for  each document in the collection 
        from in-document and in-corpus frequencies
        Args:
        coeff: contribution coefficient for in-document word frequency
               in computing word distribution
        '''
        self.distributions = []
        corpus_freq = collections.defaultdict(int)
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in self.corpus]
        num_tokens_corpus = sum(sum(x[1] for x in vec) for vec in corpus_bow)
        # iterate through documents in corpus
        for vec in corpus_bow:
            # total number of tokens (with repetitions) in current doc 
            num_tokens = sum(x[1] for x in vec)
            self.distributions.append({x[0]:coeff*x[1]/num_tokens for x in vec})
            for (word_id, count) in vec:
                # update word frequency in corpus 
                corpus_freq[word_id] += count/num_tokens_corpus

        for vec in self.distributions:
            for word_id in vec:
                # add contribution from in-corpus frequency
                vec[word_id] += (1-coeff)*corpus_freq[word_id]

    def get_distribution_given_profile(self, profile_words):
        '''
        Computes the word distribution given the apprearance of profile_words
        Args:
        profile_words: list of words that represent a discussion thread
                       (i.e., topic with all replies)
        Returns:
        Word distribution given the apprearance of profile_words
        '''
        profile_word_ids = []
        for word in profile_words:
            if word in self.dictionary.token2id:
                profile_word_ids.append(self.dictionary.token2id[word])

        #print(profile_word_ids)
        distribution = [0]*len(self.dictionary.token2id)
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
                distribution[word_id] += math.exp(log_prob)

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
        similarities = {}
        now = datetime.now()
        for date, vec in enumerate(self.dates, self.distributions):
            day_diff = (now - date).days
            similarities[topic_id] = stats.entropy(pk=distribution, qk=vec)*
                                     math.exp(day_diff/T)

        return similarities