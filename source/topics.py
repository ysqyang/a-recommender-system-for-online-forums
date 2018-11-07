# -*- coding: utf-8 -*-

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
import logging
from scipy import stats

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
    def __init__(self, topic_dict):
        self.topic_dict = topic_dict

    def get_corpus_data(self, preprocess_fn, stopwords, punc_frac_low, 
                        punc_frac_high, valid_count, valid_ratio):
        self.corpus_data = {}
        for topic_id, info in self.topic_dict.items(): 
            content = ' '.join(info['body'].split())
            word_list = preprocess_fn(content, stopwords, punc_frac_low,  
                                      punc_frac_high, valid_count, valid_ratio)

            if word_list is not None: # add only valid topics
                self.corpus_data[topic_id] = {'content': word_list,
                                              'date' = info['postDate']}

        corpus = [topic_data['content'] for topic_data in self.corpus_data.values()]
        self.dictionary = corpora.Dictionary(corpus)
        for info in self.corpus_data.values():
            info['bow'] = self.dictionary.doc2bow(info['content'])

        logging.info('%d topics available', len(self.corpus_data))

    def get_similarity_data(self, T):
        '''
        Computes the pairwise cosine similarities for the corpus 
        with time adjustments and the corresponding similarity lists
        sorted by similarity value
        '''
        self.sim_matrix = collections.defaultdict(dict)
        self.sim_sorted = collections.defaultdict(list)
        for tid_i, info_i in self.corpus_data.items():
            date_i, _ = utils.convert_timestamp(info_i['date'])
            for tid_j, info_j in self.corpus_data.items():
                date_j, _ = utils.convert_timestamp(info_j['date'])
                day_diff = abs((date_i-date_j).days)
                sim_val = matutils.cossim(info_i['bow'], info_j['bow'])*math.exp(-day_diff/T)
                self.sim_matrix[tid_i][tid_j] = sim_val

            self.sim_sorted[tid_i] = [[tid_j, sim_val] for tid_j, sim_val 
                                               in self.sim_matrix[tid_i].items()]
            self.sim_sorted[tid_i].sort(key=lambda x:x[1], reverse=True)

        logging.debug('sim_matrix_len=%d, sim_sorted_len=%d', 
                      len(self.sim_matrix), len(self.sim_sorted))

    '''
    def get_topics_by_keywords(self, keywords):
        keyword_ids = [self.dictionary.token2id[kw] for kw in keywords]
        for bow in self.corpus_bow:
    '''        

    def remove_old(self, cut_off):
        '''
        Removes all topics posted before date specified by cut_off 
        '''
        logging.info('Removing old topics from the collection...')

        delete_tids = []
        for tid, info in self.corpus_data.items():
            dt, dt_str = utils.convert_timestamp(info['date'])
            if dt  
        
        for delete_tid in delete_tids:
            del self.corpus_data[]
            del self.sim_matrix[delete_tid]
            del self.sim_sorted[delete_tid]
        
        for tid, sim_dict in self.sim_matrix.items():
            for delete_tid in delete_tids:
                del sim_dict[delete_tid]

        for tid, sim_list in self.sim_sorted.items():
            self.sim_sorted[tid] = [x for x in sim_list if x[0] not in delete_tids]
                  
        logging.debug('oldest among collection after removing: %s', new_oldest_str)
        logging.debug('latest among deleted: %s', last_cut_str)
        assert last_cut < cut_off <= new_oldest
        logging.info('Old topics removed')
        logging.info('%d topics available', len(self.corpus_data))
        logging.info('Oldest topic is now %s', self.corpus_data[0]['date'])
        logging.debug('sim_matrix_len=%d, sim_sorted_len=%d', 
                      len(self.sim_matrix), len(self.sim_sorted))
  
    def add_one(self, topic, preprocess_fn, stopwords, punc_frac_low, punc_frac_high,
                valid_count, valid_ratio, trigger_days, cut_off, T):
        content = ' '.join(topic['body'].split())
        word_list = preprocess_fn(content, stopwords, punc_frac_low,  
                                  punc_frac_high, valid_count, valid_ratio)

        #print(word_list)
        if word_list is None: # ignore invalid topics
            logging.info('Topic is not recommendable')
            return 
        
        self.dictionary.add_documents([word_list])
        new_tid, new_date_stmp = topic['topicID'], topic['postDate']
        self.corpus_data[new_tid] = {'date': topic['postDate'],
                                     'content': word_list,
                                     'bow': self.dictionary.doc2bow(word_list)}

        def sim_insert(sim_list, target_tid, target_sim_val):
            i = 0
            while i < len(sim_list) and sim_list[i][1] > target_sim_val:
                i += 1s
            sim_list.insert(i, [target_tid, target_sim_val])
             
        new_date, _ = utils.convert_timestamp(new_date_stmp)
        for tid, info in self.corpus_data.items():
            date, _ = utils.convert_timestamp(info['date'])
            day_diff = (new_date-date).days
            sim_val = matutils.cossim(self.corpus_data[new_tid]['bow'], 
                                      info['bow'])*math.exp(-day_diff/T)
            self.sim_matrix[tid][new_tid] = sim_val
            self.sim_matrix[new_tid][tid] = sim_val
            sim_insert(self.sim_sorted[tid], new_tid, sim_val)

        self.sim_sorted[new_tid] = [[tid, sim_val] for tid, sim_val 
                                    in self.sim_matrix[new_tid].items()]
        self.sim_sorted[new_tid].sort(key=lambda x:x[1], reverse=True)
       
        logging.info('New topic has been added to the collection')
        logging.info('Collection and similarity data have been updated')
        logging.info('%d topics available', len(self.corpus_data))
        logging.debug('sim_matrix_len=%d, sim_sorted_len=%d', 
                      len(self.sim_matrix), len(self.sim_sorted))

    def delete_one(self, topic_id):       
        if topic_id not in self.corpus_data:
            logging.warning('Collection is empty!')
            return 

        del self.corpus_data[topic_id]
        del self.sim_matrix[topic_id]
        del self.sim_sorted[topic_id]

        for sim_dict in self.sim_matrix.values():
            del sim_dict[topic_id]

        for tid, sim_list in self.sim_sorted.items():
            self.sim_sorted[tid] = [x for x in sim_list if x[0] != topic_id]

        logging.info('Topic %s has been deleted from the collection', topic_id)
        logging.info('Collection and similarity data have been updated')
        logging.info('%d topics remaining', len(self.corpus_data))
        logging.debug('corpus_size=%d', len(self.corpus_data))
        logging.debug('sim_matrix_len=%d, sim_sorted_len=%d', 
                      len(self.sim_matrix), len(self.sim_sorted))

    def check_correctness(self):
        '''
        Perform correctness and consistency checks
        '''
        sim_matrix_len, sim_sorted_len = len(self.sim_matrix), len(self.sim_sorted)
        assert sim_matrix_len==sim_sorted_len==len(self.corpus_data)
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

        logging.info('Similarity matrix saved to %s', sim_matrix_path)
        logging.info('Similarity lists saved to %s', sim_sorted_path)