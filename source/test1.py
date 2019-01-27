# -*- coding: utf-8 -*-

import pika
import json
import utils
from termcolor import colored
import os, sys
import jieba
import re

root_dir = os.path.dirname(sys.path[0])
config_path = os.path.abspath(os.path.join(root_dir, 'config'))
sys.path.insert(1, config_path)
import constants as const

text = '人多导致资源紧张是中国经济现实的天大谎言。现代科技和现代经济史告诉我们，资源的紧张绝非因为人口众多，而是因为生产力的落后。过去30多年，中国经济靠低廉劳动力的推动，今天中国经济，特别是制造业面临的困境，根子在于低廉劳动力的丧失。当下中国人口政策面临的最大问题绝非人口太多，生育率太高，而是在人均收入水平仍然处于中等偏下、城镇化远未完成的情况下，人口生育率过低导致人口红利提前终结，和由此带来的老龄化等一系列社会经济问题。'

stopwords = utils.load_stopwords(const._STOPWORD_FILE)

def preprocess(text): 
    alphanum, whitespace = r'\\*\w+', r'\s' 
    word_list = []
    words = jieba.cut(text, cut_all=False)
    
    for word in words:
        if len(word) == 1                                 \
           or re.match(alphanum, word, flags=re.ASCII)    \
           or re.match(whitespace, word, flags=re.ASCII)  \
           or word in stopwords                           \
           or any(c in const._SINGLES for c in word)        \
           or len(word)/len(set(word)) > 2:
            continue
        word_list.append(word) 

    return word_list

word_list = preprocess(text)

print(word_list)

