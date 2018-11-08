# -*- coding: utf-8 -*-

import re
import json
import database
from datetime import datetime
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

def convert_timestamp(timestamp):
    dt = datetime.fromtimestamp(timestamp/1000)
    dt_string = dt.strftime(const._DATETIME_FORMAT) 
    return dt, dt_string  