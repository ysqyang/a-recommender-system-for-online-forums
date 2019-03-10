# class definitions
import re
import time
from datetime import datetime, timedelta
import os
from collections import defaultdict
import math
import json
from gensim import corpora, models, matutils
#from gensim.similarities import Similarity
from gensim.models import Word2Vec
import numpy as np
import jieba


class Corpus(object):
    '''
    Corpus object
    '''
    def __init__(self, name, singles, puncs, punc_frac_low, punc_frac_high,
                 valid_count, valid_ratio, stopwords, logger):
        self.corpus_data = {}
        self.dictionary = corpora.Dictionary([])
        self.name = name
        self.singles = singles
        self.puncs = puncs
        self.punc_frac_low = punc_frac_low
        self.punc_frac_high = punc_frac_high
        self.valid_count = valid_count
        self.valid_ratio = valid_ratio
        self.stopwords = stopwords
        self.logger = logger

    def preprocess(self, text):
        '''
        Tokenize a Chinese document to a list of words
        Args:
        text: text to be tokenized
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
                                                  'body': rec['body'],
                                                  'updated': False
                                                 }
                except json.JSONDecodeError as e:
                    self.logger.error('Failed to load topic %s', file)

        self.logger.info('%d topics loaded from disk', len(self.corpus_data))

        if len(self.corpus_data) > 0:            
            corpus = [info['body'] for info in self.corpus_data.values()]
            self.dictionary = corpora.Dictionary(corpus)
            self.logger.info('Dictionary created')
            int_tids = [int(tid) for tid in self.corpus_data]
            min_tid, max_tid = min(int_tids), max(int_tids)

    def get_tfidf_model(self, scheme):
        corpus_bow = [self.dictionary.doc2bow(info['body']) 
                      for info in self.corpus_data.values()]
        return models.TfidfModel(corpus_bow, smartirs=scheme)

    def add_one(self, topic):
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
                                     'body': word_list,
                                     'updated': False}
                     
        self.logger.info('New topic has been added to collection %s', self.name)
        self.logger.info('Corpus data have been updated for collection %s', self.name)
        self.logger.info('%d topics available in collection %s', len(self.corpus_data), self.name)
        
        return True

    def delete(self, topic_id):
        topic_id = str(topic_id)
        if topic_id in self.corpus_data:
            del self.corpus_data[topic_id]
            self.logger.info('Topic %s has been deleted from the collection', topic_id)
            self.logger.info('Corpus data have been updated')
            self.logger.info('%d topics remaining', len(self.corpus_data))

    @ property
    def oldest(self):
        if len(self.corpus_data) == 0:
            return -1
        return min(self.corpus_data.keys())

    @ property
    def latest(self):
        if len(self.corpus_data) == 0:
            return -1
        return max(self.corpus_data.keys())


class Corpus_with_similarity_data(Corpus):
    '''
    Corpus collection
    '''
    def __init__(self, name, singles, puncs, punc_frac_low, punc_frac_high, 
                 valid_count, valid_ratio, stopwords, time_decay_scale,
                 duplicate_thresh, irrelevant_thresh, max_size, logger):
        super().__init__(name=name,
                         singles=singles,
                         puncs=puncs,
                         punc_frac_low=punc_frac_low,
                         punc_frac_high=punc_frac_high,
                         valid_count=valid_count,
                         valid_ratio=valid_ratio,
                         stopwords=stopwords,
                         logger=logger)
        self.sim_sorted = defaultdict(list)
        self.time_decay_scale = time_decay_scale
        self.duplicate_thresh = duplicate_thresh
        self.irrelevant_thresh = irrelevant_thresh
        self.max_size = max_size
        self.appears_in = defaultdict(list)

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
                except json.JSONDecodeError:
                    self.logger.error('Failed to load similarity data for topic %s', file)
  
    def _insert_into_sim_list(self, target_id, id_, val, max_size):
        '''
        Helper function to insert [id_, val] into the similarity list for
        target_id while maintaining the sorted order of similarity values
        '''
        l = self.sim_sorted[target_id]
        if len(l) == max_size and val < l[-1][1]:
            return False
        
        i = 0
        while i < len(l) and l[i][1] > val:
            i += 1
        
        l.insert(i, [id_, val])
        self.appears_in[id_].append(target_id)
        
        if len(l) > max_size:
            self.appears_in[l[-1][0]].remove(target_id)
            del l[-1]

        return True

    def add_one(self, topic):
        if not super().add_one(topic):
            return False

        new_tid = str(topic['topicID'])
        new_date = datetime.fromtimestamp(self.corpus_data[new_tid]['date'])
        
        bow = self.dictionary.doc2bow(self.corpus_data[new_tid]['body'])

        for tid, info in self.corpus_data.items():
            date = datetime.fromtimestamp(info['date'])
            time_factor = math.exp(-(new_date-date).days/self.time_decay_scale)
            if tid != new_tid:
                bow1 = self.dictionary.doc2bow(info['body'])
                sim_val = matutils.cossim(bow, bow1)
                sim_val_1 = sim_val * min(1, 1/time_factor)
                sim_val_2 = sim_val * min(1, time_factor)
                
                if self.irrelevant_thresh <= sim_val_1 <= self.duplicate_thresh:
                    if self._insert_into_sim_list(tid, new_tid, sim_val_1, self.max_size):
                        self.corpus_data[tid]['updated'] = True
                
                if self.irrelevant_thresh <= sim_val_2 <= self.duplicate_thresh:   
                    self._insert_into_sim_list(new_tid, tid, sim_val_2, self.max_size)

        self.logger.info('Topic %s has been added to similarity results', new_tid)
        self.logger.debug('sim_sorted_len=%d', len(self.sim_sorted))

        return True

    def delete(self, topic_id):
        print('deleting {}'.format(topic_id))
        super().delete(topic_id)

        if topic_id in self.sim_sorted:
            del self.sim_sorted[topic_id]
            self.logger.debug('sim_sorted_len=%d', len(self.sim_sorted))

        if topic_id in self.appears_in:
            for tid in self.appears_in[topic_id]: # list of topic id's whose similarity lists tid appears in
                print(self.sim_sorted[tid])
                self.sim_sorted[tid] = [x for x in self.sim_sorted[tid] if x[0]!=topic_id]
                self.corpus_data[tid]['updated'] = True

            del self.appears_in[topic_id]
            self.logger.info('Topic %s has been deleted from similarity results', topic_id)

    def remove_before(self, t):
        if type(t) != float:
            t = time.mktime(t.timetuple())
        
        for tid in list(self.corpus_data.keys()):
            if self.corpus_data[tid]['date'] < t:
                self.delete(tid)

    def find_most_similar(self, topic, n):
        '''
        Given a topic, compute its similarities with all topics 
        in the corpus and return the top n most similar ones from 
        the corpus
        '''
        sim_list = []
        bow = self.dictionary.doc2bow(topic['body'])

        for tid, info in self.corpus_data.items():
            bow1 = self.dictionary.doc2bow(info['body'])
            sim_val = matutils.cossim(bow, bow1)
            if self.irrelevant_thresh <= sim_val <= self.duplicate_thresh:
                self._insert_into_sim_list(sim_list, tid, sim_val, self.max_size)

        return sim_list

    def save(self, save_dir, mod_num):
        '''
        Saves the corpus and similarity data to disk
        Args:
        save_dir: directory under which to save the data
        mod_num:  number of data folders
        ''' 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for tid, info in self.corpus_data.items():
            if info['updated']:
                sim_record = {'date': info['date'],
                              'body': info['body'],
                              'sim_list': self.sim_sorted[tid]}
                folder_name = str(int(tid) % mod_num)
                dir_path = os.path.join(save_dir, folder_name)
                # build the subdir for storing topics
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path) 
                filename = os.path.join(dir_path, tid)
                with open(filename, 'w') as f:
                    json.dump(sim_record, f)
                info['updated'] = False
                self.logger.info('similarity data for topic %s updated on disk', tid)
            else:
                self.logger.info('No updates for topic %s', tid)