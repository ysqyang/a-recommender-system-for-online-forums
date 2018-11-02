# Class definitions for streaming data from file
from gensim import corpora, models, matutils
#from gensim.similarities import Similarity
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
                                punc_frac_low, punc_frac_high, valid_count,
                                valid_ratio, features, weights):
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
        content = ' '.join(data[str(self.topic_id)]['body'].split())
        word_list = preprocess_fn(content, stopwords)
        if word_list is None:
            self.valid = False
            return

        self.valid = True
        self.corpus.append(word_list)
        with open(self.reply_file, 'r') as f:
            # iterates through replies under this topic id       
            data = json.load(f)
        
        replies = data[str(self.topic_id)]

        for reply_id, rec in replies.items():
            content = ' '.join(rec['body'].split())
            word_list = preprocess_fn(content, stopwords, punc_frac_low, 
                                      punc_frac_high, valid_count, valid_ratio)
            if word_list is not None:
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

        tfidf = model[corpus_bow[0]]
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
    def __init__(self, topic_dict, datetime_format):
        self.topic_dict = topic_dict
        self.datetime_format = datetime_format

    def make_corpus(self, preprocess_fn, stopwords, punc_frac_low, 
                    punc_frac_high, valid_count, valid_ratio):
        # iterates through all topics
        self.corpus, self.dates, self.valid_topics, self.date_cuts = [], [], [], []
        # sorting topipc id's means sorting by post date
        sorted_topic_ids = sorted(list(self.topic_dict.keys()))

        #last_date = datetime.strptime('2018-01-01 00:00:00', self.datetime_format)
        for topic_id in sorted_topic_ids:
            topic = self.topic_dict[topic_id]
            post_date, content = topic['POSTDATE'], topic['body'] 
            content = ' '.join(content.split())
            word_list = preprocess_fn(content, stopwords, punc_frac_low,  
                                      punc_frac_high, valid_count, valid_ratio)

            if word_list is not None: # add only valid topics
                self.corpus.append(word_list)
                self.valid_topics.append(topic_id)
                #cur_date = datetime.strptime(post_date, self.datetime_format)
                #if cur_date.date() > last_date.date():
                #    self.cuts.append(len(self.dates))
                self.dates.append(post_date)
                #last_date = cur_date

    def get_bow(self):
        self.dictionary = corpora.Dictionary(self.corpus)
        self.corpus_bow = [self.dictionary.doc2bow(doc) for doc in self.corpus]

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
        num_tokens_corpus = sum(sum(x[1] for x in vec) for vec in self.corpus_bow)

        for i, vec in enumerate(self.corpus_bow):
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
        for v, log_prob in zip(self.distributions, log_probs):
            #print(log_prob)
            for wid in range(len(v)): 
                if wid not in profile_wids:
                    distribution[wid] += math.exp(log_prob+math.log(v[wid]))

        # normalize the probabilities
        s = sum(distribution)
        for i in range(len(distribution)):
            distribution[i] /= s
        
        return distribution

    def get_similarity_given_distribution(self, distribution, T):
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
                date = datetime.strptime(date, self.datetime_format)
                day_diff = (now - date).days
                sim[tid] = stats.entropy(pk=distribution, qk=vec)*math.exp(day_diff/T)
        return sim

    def get_similarity_data(self, T):
        '''
        Computes the pairwise cosine similarities for the corpus 
        with time adjustments and the corresponding similarity lists
        sorted by similarity value
        '''
        self.sim_matrix = collections.defaultdict(dict)
        self.sim_sorted = collections.defaultdict(list)
        for i, tid_i in enumerate(self.valid_topics):
            vec_i = self.corpus_bow[i]
            date_i = datetime.strptime(self.dates[i], self.datetime_format)
            for j, tid_j in enumerate(self.valid_topics):
                vec_j = self.corpus_bow[j]
                date_j = datetime.strptime(self.dates[j], self.datetime_format)
                day_diff = abs((date_i-date_j).days)
                self.sim_matrix[tid_i][tid_j] = matutils.cossim(vec_i, vec_j)*math.exp(-day_diff/T)

            self.sim_sorted[tid_i] = [[tid_j, sim_val] for tid_j, sim_val 
                                      in self.sim_matrix[tid_i].items()]
            self.sim_sorted[tid_i].sort(key=lambda x:x[1], reverse=True)

    '''
    def get_topics_by_keywords(self, keywords):
        keyword_ids = [self.dictionary.token2id[kw] for kw in keywords]
        for bow in self.corpus_bow:
    '''        
    def remove_old(self, cut_off):
        '''
        Removes all topics posted before date specified by cut_off 
        '''
        oldest = datetime.strptime(self.dates[0], self.datetime_format)
        latest = datetime.strptime(self.dates[-1], self.datetime_format)

        if cut_off < oldest or cut_off > latest:
            return 
        #binary search for the first entry later than cut_off
        l, r = 0, len(self.dates)-1
        while l < r:
            mid = (l+r)//2
            mid_date = datetime.strptime(self.dates[mid], self.datetime_format)
            if mid_date.date() > cut_off.date():
                r = mid
            elif mid_date.date() <= cut_off.date():
                l = mid+1
            
        del self.corpus[:l]       
        del self.dates[:l]
        del self.corpus_bow[:l]
        delete_tids = set(self.valid_topics[:l])
        del self.valid_topics[:l]

        for delete_tid in delete_tids:
            del self.sim_matrix[delete_tid]
            del self.sim_sorted[delete_tid]
        
        for tid, sim_dict in self.sim_matrix.items():
            for delete_tid in delete_tids:
                del sim_dict[delete_tid]

        for tid, sim_list in self.sim_sorted.items():
            self.sim_sorted[tid] = [x for x in sim_list if x[0] not in delete_tids]
  
    def add_one(self, topic, preprocess_fn, stopwords, punc_frac_low, 
                punc_frac_high, valid_count, valid_ratio, trigger_days, cut_off, T):
        content = ' '.join(topic['body'].split())
        word_list = preprocess_fn(content, stopwords, punc_frac_low,  
                                  punc_frac_high, valid_count, valid_ratio)

        #print(word_list)
        if word_list is None: # ignore invalid topics
            return -1

        new_tid = topic['topicid']
        new_bow = self.dictionary.doc2bow(word_list)
        new_date = datetime.strptime(topic['POSTDATE'], self.datetime_format)

        def insert(tid, target_tid, target_sim_val):
            sim_list = self.sim_sorted[tid]
            
            if len(sim_list) == 0:
                sim_list.append([target_tid, target_sim_val])
                return

            l, r = 0, len(sim_list)
            while l < r:
                m = (l+r)//2
                if sim_list[m][1] <= target_sim_val:
                    r = m
                else:
                    l = m+1 

            sim_list.insert(l, [target_tid, target_sim_val])
      
        self.sim_matrix[new_tid][new_tid] = 1.0       
        for tid, bow, date in zip(self.valid_topics, self.corpus_bow, self.dates):
            date = datetime.strptime(date, self.datetime_format)
            day_diff = (new_date-date).days
            sim_val = matutils.cossim(new_bow, bow)*math.exp(-day_diff/T)
            self.sim_matrix[tid][new_tid] = sim_val
            self.sim_matrix[new_tid][tid] = sim_val
            insert(tid, new_tid, sim_val)

        self.sim_sorted[new_tid] = [[tid_j, sim_val] for tid_j, sim_val 
                                    in self.sim_matrix[new_tid].items()]
        self.sim_sorted[new_tid].sort(key=lambda x:x[1], reverse=True)

        self.corpus.append(word_list)
        self.dates.append(topic['POSTDATE'])
        self.valid_topics.append(topic['topicid'])
        self.dictionary.add_documents([word_list])
        self.corpus_bow.append(new_bow)
        #print('oldest: ', self.dates[0])
        #print('latest: ', self.dates[-1])

        return 0

    def delete_one(self, topic_id):
        def bin_search(topic_id):
            topic_list = self.valid_topics
            l, r = 0, len(topic_list)-1
            while l < r:
                m = (l+r)//2
                if int(topic_list[m]) < int(topic_id):
                    l = m+1
                elif int(topic_list[m]) > int(topic_id):
                    r = m
                else:
                    return m 

            return l if topic_list[l] == topic_id else -1
        
        idx = bin_search(topic_id)
        if idx >= 0:
            del self.corpus[idx]
            del self.corpus_bow[idx]
            del self.dates[idx]
            del self.valid_topics[idx]
            del self.sim_matrix[topic_id]
            del self.sim_sorted[topic_id]

            for tid, sim_dict in self.sim_matrix.items():
                del sim_dict[topic_id]

            for tid, sim_list in self.sim_sorted.items():
                self.sim_sorted[tid] = [x for x in sim_list if x[0] != topic_id]

    def check_correctness(self):
        '''
        Perform correctness and consistency checks
        '''
        corpus_size = len(self.corpus)
        assert corpus_size==len(self.valid_topics)==len(self.dates)==len(self.corpus_bow)
        sim_matrix_len, sim_sorted_len = len(self.sim_matrix), len(self.sim_sorted)
        assert sim_matrix_len==sim_sorted_len==corpus_size
        assert all(len(sim_dict)==sim_matrix_len for tid, sim_dict in self.sim_matrix.items())
        assert all(len(sim_list)==sim_sorted_len for tid, sim_list in self.sim_sorted.items())

    def save_similarity_data(self, sim_matrix_path, sim_sorted_path):
        '''
        Saves the similarity matrix and sorted similarity lists
        to disk
        Args:
        sim_matrix_path: file path for similarity matrix
        sim_sorted_path: file path for sorted similarity lists
        '''
        self.check_correctness() # always check correctness before saving
        with open(sim_matrix_path, 'w') as f:
            json.dump(self.sim_matrix, f)

        with open(sim_sorted_path, 'w') as f:
            json.dump(self.sim_sorted, f)