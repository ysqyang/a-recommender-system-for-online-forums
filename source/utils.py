# -*- coding: utf-8 -*-

import jieba
import re
import json
import database
import datetime
import constants as const
import configparser
import logging

def load_stopwords(stopwords_path):
    stopwords = set()
    with open(stopwords_path, 'r') as f:
        n = 1
        while True:
            stopword = f.readline()
            if stopword == '':
                break
            stopwords.add(stopword.strip('\n'))
            n += 1

    logging.info('Stopwords loaded to memory')
    return stopwords

def save_topics(topic_dict, path):
    with open(path, 'w') as f:
        json.dump(topic_dict, f)
    logging.info('topic_dict saved to %s', path)

def get_config(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    logging.info('Configuration read')
    return config

def preprocess(text, stopwords, punc_frac_low, punc_frac_high, 
               valid_count, valid_ratio):
    '''
    Tokenize a Chinese document to a list of words and filters out
    invalid documents 
    Args:
    text:            text to be tokenized
    stopwords:       set of stopwords
    punc_ratio_low:  lower threshold for the fraction of punctuation marks
    punc_ratio_high: upper threshold for the fraction of punctuation marks
    valid_count:     lower limit of the number of tokens
    valid_ratio:     lower threshold for the ratio of token count to 
                     distinct token count  
    '''  
    puncs = {'。', '，', '、', '：', ':', ';', '；', '“', '”', ' '}
    cnt = 0
    for c in text:
        if c in puncs:
            cnt += 1
    
    ratio = cnt / len(text)

    if ratio < punc_frac_low or ratio > punc_frac_high:
        return None

    singles = {'一', '二', '三', '四', '五',
              '六', '七', '八', '九', '十', 
              '两', '这', '那', '不', '很',
              '是', '只', '就', '你', '我', 
              '他', '她', '它', '啊', '呵',
              '哈', '哦'}

    alphanum, whitespace = r'\\*\w+', r'\s' 
    word_list = []
    words = jieba.cut(text, cut_all=False)
    
    for word in words:
        if re.match(alphanum, word, flags=re.ASCII):
            continue
        if re.match(whitespace, word, flags=re.ASCII):
            continue
        if word in stopwords or any(c in singles for c in word) :
            continue
        if len(word)/len(set(word)) > 2: 
            continue
        word_list.append(word) 

    if len(word_list) < valid_count:
        return None

    if len(word_list)/len(set(word_list)) > valid_ratio:
        return None

    return word_list   