# class definitions
from gensim import corpora, models, matutils
#from gensim.similarities import Similarity
from sklearn import preprocessing
import warnings
import numpy as np
from collections import defaultdict
import math
import json
import jieba
from scipy import stats
import re
from datetime import datetime, timedelta
import os

class Corpus(object):
    '''
    Corpus object
    '''
    def __init__(self, singles, stopwords, trigger_days, 
                 keep_days, T, logger):
        self.corpus_data = {}
        self.oldest = datetime.max
        self.latest = datetime.min
        self.singles = singles
        self.stopwords = stopwords
        self.trigger_days = trigger_days
        self.keep_days = keep_days
        self.T = T
        self.logger = logger

    def preprocess(self, text):
        '''
        Tokenize a Chinese document to a list of words
        Args:
        text:            text to be tokenized 
        '''  
        cnt = 0
        for c in text:
            if c in self.puncs:
                cnt += 1
        
        ratio = cnt / len(text)

        if ratio < self.punc_frac_low or ratio > self.punc_frac_high:
            return []

        alphanum, whitespace = r'\\*\w+', r'\s' 
        word_list = []
        words = jieba.cut(text, cut_all=False)
        
        for word in words:
            if len(word) == 1                                 \
               or re.match(alphanum, word, flags=re.ASCII)    \
               or re.match(whitespace, word, flags=re.ASCII)  \
               or word in self.stopwords                      \
               or any(c in self.singles for c in word)        \
               or len(word)/len(set(word)) > 2:
                continue
            word_list.append(word) 

        if len(word_list) < self.valid_count                          \
           or len(word_list)/len(set(word_list)) > self.valid_ratio:  \
            return []

        return word_list

    def get_dictionary(self):
        corpus = [info['content'] for info in self.corpus_data.values()]
        self.dictionary = corpora.Dictionary(corpus)

    def load(self, save_dir):       
        folders = os.listdir(save_dir)
        for folder in folders:
            if not folder.isnumeric():
                continue
            files = os.listdir(os.path.join(save_dir, folder))
            for file in files:
                if not file.isnumeric():
                    continue
                path = os.path.join(save_dir, folder, file)
                try:
                    with open(path, 'r') as f:
                        rec = json.load(f)
                        self.corpus_data[file] = {'date': rec['date'],
                                                  'content': rec['content'],
                                                  'bow': rec['bow'],
                                                  'updated': False
                                                 }
                except json.JSONDecodeError as e:
                    self.logger.error('Failed to load topic %s', file)

        self.logger.info('%d topics loaded from disk', len(self.corpus_data))
        if len(self.corpus_data) > 0:            
            self.get_dictionary()
            int_tids = [int(tid) for tid in self.corpus_data]
            min_tid, max_tid = min(int_tids), max(int_tids)
            self.oldest = datetime.fromtimestamp(self.corpus_data[str(min_tid)]['date'])
            self.latest = datetime.fromtimestamp(self.corpus_data[str(max_tid)]['date'])
  
    def get_tfidf_model(self):
        corpus_bow = [info['bow'] for info in self.corpus_data.values()]
        return models.TfidfModel(corpus_bow, smartirs=self.scheme) 

    def remove_old(self):
        '''
        Removes all topics posted before date specified by cut_off 
        '''
        #print(self.oldest, self.latest)
        if (self.latest - self.oldest).days <= self.trigger_days:
            self.logger.info('No removal needed')
            return []
        
        cut_off = self.latest - timedelta(days=self.keep_days)
        self.logger.info('Removing old topics from the collection...')
        delete_tids = []
        for tid, info in self.corpus_data.items():
            if datetime.fromtimestamp(info['date']) < cut_off:
                delete_tids.append(tid)

        for delete_tid in delete_tids:
            del self.corpus_data[delete_tid]
        
        min_tid = min(int(tid) for tid in self.corpus_data)
        oldest_stmp = self.corpus_data[str(min_tid)]['date'] 
        self.oldest = datetime.fromtimestamp(oldest_stmp)
        self.logger.info('%d topics available', len(self.corpus_data))
        assert (self.latest - self.oldest).days <= self.keep_days
        
        return delete_tids

    def add_one(self, topic):
        new_date = datetime.fromtimestamp(topic['postDate'])
        if (self.latest - new_date).days > self.trigger_days:
            self.logger.info('Topic not in date range')
            return False

        new_tid = str(topic['topicID'])
        if new_tid in self.corpus_data:
            self.logger.warning('Topic already exists. Ignoring...')
            return False

        word_list = self.preprocess(' '.join(topic['body'].split()))
        if len(word_list) == 0:
            self.logger.info('Topic not recommendable')
            return False      
        
        self.dictionary.add_documents([word_list])
        bow = self.dictionary.doc2bow(word_list)

        self.corpus_data[new_tid] = {'date': topic['postDate'],
                                     'content': word_list,
                                     'bow': bow,
                                     'updated': True}

        self.oldest = min(self.oldest, new_date)
        self.latest = max(self.latest, new_date)
                     
        self.logger.info('New topic has been added to the collection')
        self.logger.info('Corpus data have been updated')
        self.logger.info('%d topics available', len(self.corpus_data))
        return True

    def delete_one(self, topic_id):       
        topic_id = str(topic_id)
        if topic_id not in self.corpus_data:
            self.logger.warning('Topic not found in collection')
            return False

        delete_date = datetime.fromtimestamp(self.corpus_data[topic_id]['date'])

        del self.corpus_data[topic_id]
        
        if len(self.corpus_data) == 0:
            self.oldest, self.latest = datetime.max, datetime.min
        else:    
            if delete_date == self.oldest:
                min_tid = min(int(tid) for tid in self.corpus_data)
                oldest_stmp = self.corpus_data[str(min_tid)]['date']
                self.oldest = datetime.fromtimestamp(oldest_stmp)
            if delete_date == self.latest:
                max_tid = max(int(tid) for tid in self.corpus_data)
                latest_stmp = self.corpus_data[str(max_tid)]['date']
                self.latest = datetime.fromtimestamp(latest_stmp)

        self.logger.info('Topic %s has been deleted from the collection', topic_id)
        self.logger.info('Corpus data have been updated')
        self.logger.info('%d topics remaining', len(self.corpus_data))

        return True

class Corpus_with_similarity_data(Corpus):
    '''
    Corpus collection
    '''
    def __init__(self, puncs, singles, stopwords, punc_frac_low, punc_frac_high, 
                 valid_count, valid_ratio, trigger_days, keep_days, T, 
                 duplicate_thresh, irrelevant_thresh, max_size, logger):
        super().__init__(singles, stopwords, trigger_days, keep_days, T, logger)
        self.sim_sorted = defaultdict(list)
        self.puncs = puncs
        self.punc_frac_low = punc_frac_low
        self.punc_frac_high = punc_frac_high
        self.valid_count = valid_count
        self.valid_ratio = valid_ratio
        self.duplicate_thresh = duplicate_thresh
        self.irrelevant_thresh = irrelevant_thresh
        self.max_size = max_size

    def load(self, save_dir):       
        super().load(save_dir)
        
        folders = os.listdir(save_dir)
        for folder in folders:
            if not folder.isnumeric():
                continue
            files = os.listdir(os.path.join(save_dir, folder))
            for file in files:
                if not file.isnumeric():
                    continue
                path = os.path.join(save_dir, folder, file)
                try:
                    with open(path, 'r') as f:
                        rec = json.load(f)
                        self.sim_sorted[file] = rec['sim_list']
                except json.JSONDecodeError as e:
                    self.logger.error('Failed to load similarity data for topic %s', file)
            
    def remove_old(self):
        '''
        Removes all topics posted before date specified by cut_off 
        '''
        delete_tids = super().remove_old()
        if delete_tids == []:
            return delete_tids
        
        for delete_tid in delete_tids:
            if delete_tid in self.sim_sorted:
                del self.sim_sorted[delete_tid]

        for tid, sim_list in self.sim_sorted.items():
            self.sim_sorted[tid] = [x for x in sim_list if x[0] not in delete_tids]

        return delete_tids
  
    def add_one(self, topic):
        if not super().add_one(topic):
            return False

        def sim_insert(sim_list, target_tid, target_sim_val):
            i = 0
            while i < len(sim_list) and sim_list[i][1] > target_sim_val:
                i += 1
            if i == len(sim_list) == self.max_size:
                return False
            sim_list.insert(i, [target_tid, target_sim_val])
            if len(sim_list) > self.max_size:
                del sim_list[self.max_size:]  
            return True       

        new_tid = str(topic['topicID'])
        new_date = datetime.fromtimestamp(self.corpus_data[new_tid]['date'])
        bow = self.corpus_data[new_tid]['bow']
        self.sim_sorted[new_tid] = []

        for tid, info in self.corpus_data.items():
            date = datetime.fromtimestamp(info['date'])
            time_factor = math.exp(-(new_date-date).days/self.T)
            if tid != new_tid:
                sim_val = matutils.cossim(bow, info['bow'])
                sim_val_1 = sim_val * min(1, 1/time_factor)
                sim_val_2 = sim_val * min(1, time_factor)
                if self.irrelevant_thresh <= sim_val_1 <= self.duplicate_thresh:
                    if sim_insert(self.sim_sorted[tid], new_tid, sim_val_1):
                        self.corpus_data[tid]['updated'] = True
                if self.irrelevant_thresh <= sim_val_2 <= self.duplicate_thresh:   
                    sim_insert(self.sim_sorted[new_tid], tid, sim_val_2)

        self.logger.info('Topic %s has been added to similarity results', new_tid)
        self.logger.debug('sim_sorted_len=%d', len(self.sim_sorted))

        return True

    def delete_one(self, topic_id):       
        if not super().delete_one(topic_id):
            return False

        if topic_id in self.sim_sorted:
            del self.sim_sorted[topic_id]

        for tid, sim_list in self.sim_sorted.items():
            self.sim_sorted[tid] = [x for x in sim_list if x[0] != topic_id]

        self.logger.info('Topic %s has been deleted from similarity results', topic_id)
        self.logger.debug('sim_sorted_len=%d', len(self.sim_sorted))

        return True

    def save(self, save_dir, mod_num):
        '''
        Saves the corpus and similarity data to disk
        Args:
        save_dir: directory under which to save the data
        mod_num:  number of data folders
        ''' 
        assert len(self.sim_sorted)==len(self.corpus_data)
        for tid, info in self.corpus_data.items():
            if info['updated']:
                sim_record = {'date': info['date'],
                              'content': info['content'],
                              'bow': info['bow'], 
                              'sim_list': self.sim_sorted[tid]}
                folder_name = str(int(tid) % mod_num)
                folder_path = os.path.join(save_dir, folder_name)
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                filename = os.path.join(folder_path, tid)
                with open(filename, 'w') as f:
                    json.dump(sim_record, f)
                info['updated'] = False
                self.logger.info('similarity data for topic %s updated on disk', tid)
            else:
                self.logger.info('No updates for topic %s', tid)