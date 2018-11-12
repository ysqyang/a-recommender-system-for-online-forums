# -*- coding: utf-8 -*-

# Class definitions for streaming data from file
from gensim import corpora, models, matutils
#from gensim.similarities import Similarity
from sklearn import preprocessing
import warnings
import numpy as np
from collections import defaultdict
import math
import json
import jieba
import logging
from scipy import stats
import re
from datetime import datetime, timedelta

class Topic_collection(object):
    '''
    Corpus collection
    '''
    def __init__(self, puncs, singles, stopwords, punc_frac_low, 
                 punc_frac_high, valid_count, valid_ratio, 
                 trigger_days, keep_days, T, irrelevant_thresh):
        self.corpus_data = {}
        self.sim_matrix = defaultdict(dict)
        self.sim_sorted = defaultdict(list)
        self.oldest = datetime.max
        self.latest = datetime.min
        self.puncs = puncs
        self.singles = singles
        self.stopwords = stopwords
        self.punc_frac_low = punc_frac_low
        self.punc_frac_high = punc_frac_high
        self.valid_count = valid_count
        self.valid_ratio = valid_ratio
        self.trigger_days = trigger_days
        self.keep_days = keep_days
        self.T = T
        self.irrelevant_thresh = irrelevant_thresh

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
        assert len(self.sim_matrix)==len(self.sim_sorted)==len(self.corpus_data)
        assert all(tid in self.sim_sorted for tid in self.sim_matrix)
        assert all(tid in self.sim_matrix for tid in self.sim_sorted)
        assert all(len(self.sim_matrix[tid])==len(self.sim_sorted[tid]) for tid in self.sim_matrix)

    def get_dictionary(self):
        corpus = [info['content'] for info in self.corpus_data.values()]
        self.dictionary = corpora.Dictionary(corpus)

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
        int_tids = [int(tid) for tid in self.corpus_data]
        min_tid, max_tid = min(int_tids), max(int_tids)
        self.oldest = datetime.fromtimestamp(self.corpus_data[str(min_tid)]['date'])
        self.latest = datetime.fromtimestamp(self.corpus_data[str(max_tid)]['date'])

    def get_topics_by_keywords(self, keyword_weight):
        recoms = defaultdict(int)
        for tid, info in self.corpus_data.items():
            for word_id, freq in info['bow']:
                word = self.dictionary[word_id]
                if word in keyword_weight:
                    recoms[tid] += freq*keyword_weight[word]

        return sorted(recoms.items(), key=lambda x:x[1], reverse=True)
            
    def remove_old(self, cut_off):
        '''
        Removes all topics posted before date specified by cut_off 
        '''
        logging.info('Removing old topics from the collection...')
        delete_tids = []
        for tid, info in self.corpus_data.items():
            if datetime.fromtimestamp(info['date']) < cut_off:
                delete_tids.append(tid)

        for delete_tid in delete_tids:
            del self.corpus_data[delete_tid]
            del self.sim_matrix[delete_tid]
            del self.sim_sorted[delete_tid]
        
        for sim_dict in self.sim_matrix.values():
            for delete_tid in delete_tids:
                if delete_tid in sim_dict:
                    del sim_dict[delete_tid]

        for tid, sim_list in self.sim_sorted.items():
            self.sim_sorted[tid] = [x for x in sim_list if x[0] not in delete_tids]
        
        min_tid = min(int(tid) for tid in self.corpus_data)
        oldest_stmp = self.corpus_data[str(min_tid)]['date'] 
        self.oldest = datetime.fromtimestamp(oldest_stmp)
        logging.info('%d topics available', len(self.corpus_data))
        logging.debug('sim_matrix_len=%d, sim_sorted_len=%d', 
                      len(self.sim_matrix), len(self.sim_sorted))
  
    def add_one(self, topic):
        new_date = datetime.fromtimestamp(topic['postDate'])
        if (self.latest - new_date).days > self.trigger_days:
            logging.info('Topic is not in date range')
            return False

        word_list = self.preprocess(' '.join(topic['body'].split()))
        if word_list is None: # ignore invalid topics
            logging.info('Topic is not recommendable')
            return False

        self.dictionary.add_documents([word_list])
        new_tid = str(topic['topicID'])
        bow = self.dictionary.doc2bow(word_list)

        self.corpus_data[new_tid] = {'date': topic['postDate'],
                                     'content': word_list,
                                     'bow': bow}

        self.oldest = min(self.oldest, new_date)
        self.latest = max(self.latest, new_date)

        if (self.latest - self.oldest).days > self.trigger_days:
            self.remove_old(self.latest - timedelta(days=self.keep_days))

        def sim_insert(sim_list, target_tid, target_sim_val):
            i = 0
            while i < len(sim_list) and sim_list[i][1] > target_sim_val:
                i += 1
            sim_list.insert(i, (target_tid, target_sim_val))           
        
        for tid, info in self.corpus_data.items():
            date = datetime.fromtimestamp(info['date'])
            time_factor = math.exp(-(new_date-date).days/self.T)
            if tid != new_tid:
                sim_val = matutils.cossim(bow, info['bow'])
                if sim_val >= self.irrelevant_thresh:
                    self.sim_matrix[tid][new_tid] = sim_val*min(1, 1/time_factor)
                    self.sim_matrix[new_tid][tid] = sim_val*min(1, time_factor)
                    sim_insert(self.sim_sorted[tid], new_tid, self.sim_matrix[tid][new_tid])

        self.sim_sorted[new_tid] = sorted(self.sim_matrix[new_tid].items(), 
                                          key=lambda x:x[1], reverse=True)
                     
        assert (self.latest - self.oldest).days <= self.trigger_days
        logging.info('New topic has been added to the collection')
        logging.info('Collection and similarity data have been updated')
        logging.info('%d topics available', len(self.corpus_data))
        logging.debug('sim_matrix_len=%d, sim_sorted_len=%d', 
                      len(self.sim_matrix), len(self.sim_sorted))

        return True

    def delete_one(self, topic_id):       
        topic_id = str(topic_id)
        if topic_id not in self.corpus_data:
            logging.warning('Topic not found in collection')
            return False 

        delete_date = datetime.fromtimestamp(self.corpus_data[topic_id]['date'])

        del self.corpus_data[topic_id]
        del self.sim_matrix[topic_id]
        del self.sim_sorted[topic_id]

        for sim_dict in self.sim_matrix.values():
            if topic_id in sim_dict:
                del sim_dict[topic_id]

        for tid, sim_list in self.sim_sorted.items():
            self.sim_sorted[tid] = [x for x in sim_list if x[0] != topic_id]

        if len(self.corpus_data) == 0:
            self.oldest, self.latest = datetime.max, datetime.min
        else:    
            int_tids = [int(tid) for tid in self.corpus_data]
            min_tid, max_tid = min(int_tids), max(int_tids)
            if delete_date == self.oldest:
                oldest_stmp = self.corpus_data[str(min_tid)]['date']
                self.oldest = datetime.fromtimestamp(oldest_stmp)
            if delete_date == self.latest:
                latest_stmp = self.corpus_data[str(max_tid)]['date']
                self.latest = datetime.fromtimestamp(latest_stmp)

        logging.info('Topic %s has been deleted from the collection', topic_id)
        logging.info('Collection and similarity data have been updated')
        logging.info('%d topics remaining', len(self.corpus_data))
        logging.debug('corpus_size=%d', len(self.corpus_data))
        logging.debug('sim_matrix_len=%d, sim_sorted_len=%d', 
                      len(self.sim_matrix), len(self.sim_sorted))

        return True

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