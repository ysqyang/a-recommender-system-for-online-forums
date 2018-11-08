# -*- coding: utf-8 -*-

import json
import database
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

def get_config(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    logging.info('Configuration loaded')
    return config
  