# -*- coding: utf-8 -*-

import logging
import os

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

def get_mq_config(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    logging.info('Configuration loaded')
    return config

def get_logger_with_config(name, logger_level, handler_levels, 
                           log_dir, mode, log_format):
    logger = logging.getLogger(name)
    
    logger.setLevel(logger_level)

    formatter = logging.Formatter(log_format)
    for level in handler_levels:
        filename = os.path.join(log_dir, '{}.{}'.format(name, level))
        handler = logging.FileHandler(filename=filename, mode=mode)
        handler.setLevel(handler_levels[level])
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def get_logger(name):
    return logging.getLogger(name)
  