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
import jieba
import constants as const
import logging
from scipy import stats

class Topic_collection(object):
    '''
    Corpus collection
    '''
    def __init__(self, puncs, singles, stopwords, punc_frac_low, 
                 punc_frac_high, valid_count, valid_ratio, T):
        self.corpus_data = {}
        self.sim_matrix = collections.defaultdict(dict)
        self.sim_sorted = collections.defaultdict(list)
        self.puncs = puncs
        self.singles = singles
        self.stopwords = stopwords
        self.punc_frac_low = punc_frac_low
        self.punc_frac_high = punc_frac_high
        self.valid_count = valid_count
        self.valid_ratio = valid_ratio
        self.T = T

    def preprocess(self, text):
        '''
        Tokenize a Chinese document to a list of words and filters out
        invalid documents 
        Args:
        text:            text to be tokenized 
        '''  
        cnt = 0
        for c in text:
            if c in self.puncs:
                cnt += 1
        
        ratio = cnt / len(text)

        if ratio < self.punc_frac_low or ratio > self.punc_frac_high:
            return None

        alphanum, whitespace = r'\\*\w+', r'\s' 
        word_list = []
        words = jieba.cut(text, cut_all=False)
        
        for word in words:
            if re.match(alphanum, word, flags=re.ASCII):
                continue
            if re.match(whitespace, word, flags=re.ASCII):
                continue
            if word in self.stopwords or any(c in self.singles for c in word) :
                continue
            if len(word)/len(set(word)) > 2: 
                continue
            word_list.append(word) 

        if len(word_list) < self.valid_count:
            return None

        if len(word_list)/len(set(word_list)) > self.valid_ratio:
            return None

        return word_list 

    def check_correctness(self):
        '''
        Perform correctness and consistency checks
        '''
        sim_matrix_len, sim_sorted_len = len(self.sim_matrix), len(self.sim_sorted)
        assert sim_matrix_len==sim_sorted_len==len(self.corpus_data)
        assert all(len(sim_dict)==sim_matrix_len for tid, sim_dict in self.sim_matrix.items())
        assert all(len(sim_list)==sim_sorted_len for tid, sim_list in self.sim_sorted.items())

    def get_dictionary(self):
        corpus = [info['content'] for info in self.corpus_data.values()]
        self.dictionary = corpora.Dictionary(corpus)

    def get_oldest(self):
        if len(self.corpus_data) > 0:
            oldest_stmp = self.corpus_data[min(self.corpus_data.keys())]['date']
            oldest, _ = utils.convert_timestamp(oldest_stmp)
            return oldest
        
    def get_latest(self):
        if len(self.corpus_data) > 0:
            latest_stmp = self.corpus_data[min(self.corpus_data.keys())]['date']
            latest, _ = utils.convert_timestamp(latest_stmp)
            return latest 

    def load(self, corpus_data_path, sim_matrix_path, sim_sorted_path):       
        with open(corpus_data_path, 'r') as f1, \
             open(sim_matrix_path, 'r') as f2,  \
             open(sim_sorted_path, 'r') as f3:  
            
            self.corpus_data = json.load(f1)
            logging.info('Corpus data loaded to memory')      
            self.sim_matrix = json.load(f2)
            logging.info('Similarity matrix loaded to memory') 
            self.sim_sorted = json.load(f3)
            logging.info('Similarity lists loaded to memory')

        self.check_correctness()
        logging.info('%d topics available', len(self.corpus_data))
        self.get_dictionary()

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
            if dt < cut_off:
                delete_tids.append(tid)

        for delete_tid in delete_tids:
            del self.corpus_data[delete_tid]
            del self.sim_matrix[delete_tid]
            del self.sim_sorted[delete_tid]
        
        for sim_dict in self.sim_matrix.values():
            for delete_tid in delete_tids:
                del sim_dict[delete_tid]

        for tid, sim_list in self.sim_sorted.items():
            self.sim_sorted[tid] = [x for x in sim_list if x[0] not in delete_tids]
        
        logging.info('%d topics available', len(self.corpus_data))
        logging.debug('sim_matrix_len=%d, sim_sorted_len=%d', 
                      len(self.sim_matrix), len(self.sim_sorted))
  
    def add_one(self, topic):
        word_list = self.preprocess(' '.join(topic['body'].split()))
        if word_list is None: # ignore invalid topics
            logging.info('Topic is not recommendable')
            return 
        
        self.dictionary.add_documents([word_list])
        new_tid, new_date_stmp = str(topic['topicID']), topic['postDate']
        self.corpus_data[new_tid] = {'date': topic['postDate'],
                                     'content': word_list,
                                     'bow': self.dictionary.doc2bow(word_list)}

        def sim_insert(sim_list, target_tid, target_sim_val):
            i = 0
            while i < len(sim_list) and sim_list[i][1] > target_sim_val:
                i += 1
            sim_list.insert(i, [target_tid, target_sim_val])
             
        new_date, _ = utils.convert_timestamp(new_date_stmp)
        for tid, info in self.corpus_data.items():
            date, _ = utils.convert_timestamp(info['date'])
            day_diff = (new_date-date).days
            sim_val = matutils.cossim(self.corpus_data[new_tid]['bow'], 
                                      info['bow'])*math.exp(-day_diff/self.T)
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

    def save(self, corpus_data_path, sim_matrix_path, sim_sorted_path):
        '''
        Saves the similarity matrix and sorted similarity lists
        to disk
        Args:
        sim_matrix_path: file path for similarity matrix
        sim_sorted_path: file path for sorted similarity lists
        '''
        self.check_correctness() # always check correctness before saving
        with open(corpus_data_path, 'w') as f1, \
             open(sim_matrix_path, 'w') as f2,  \
             open(sim_sorted_path, 'w') as f3:
            json.dump(self.corpus_data, f1)
            logging.info('Corpus data saved to %s', corpus_data_path)
            json.dump(self.sim_matrix, f2)
            logging.info('Similarity matrix saved to %s', sim_matrix_path)
            json.dump(self.sim_sorted, f3)
            logging.info('Similarity lists saved to %s', sim_sorted_path)