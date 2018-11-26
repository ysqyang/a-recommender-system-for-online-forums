# -*- coding: utf-8 -*-
import os
import sys

_ROOT               = os.path.dirname(sys.path[0])
_RESULT_DIR         = os.path.join(_ROOT, 'results')
_DATA_DIR           = os.path.join(_ROOT, 'data')
_STOPWORD_FILE      = os.path.join(_ROOT, 'stopwords.txt')
_TOPIC_FILE         = os.path.join(_DATA_DIR, 'topics')
_REPLY_FILE         = os.path.join(_DATA_DIR, 'replies') 
#_PROFILES          = os.path.join(_RESULTS_DIR, 'profiles')
#_PROFILE_WORDS     = os.path.join(_RESULTS_DIR, 'profile_words')
_SPECIAL_DIR        = os.path.join(_RESULT_DIR, 'special_topics')
_TOPIC_DIR          = os.path.join(_RESULT_DIR, 'topics')
_RECOM_DIR          = os.path.join(_RESULT_DIR, 'recoms')
_SMARTIRS_SCHEME    = 'ntn'
_SLEEP_TIME         = 10
_DB_INFO            = ('192.168.1.102','tgbweb','tgb123321','taoguba', 3307, 'utf8mb4')
_EXCHANGE_NAME      = 'recommender'
_DATETIME_FORMAT    = '%Y-%m-%d %H:%M:%S'
_NUM_RESULT_DIRS    = 1000
_SAVE_INTERVAL      = 30     # number of seconds between successive saves
_TIMESTAMP_FACTOR   = 1000
_DAYS               = 90
_T                  = 30     #time decay factor
_MIN_LEN            = 90
_MIN_REPLIES        = 0
_MIN_REPLIES_1      = 20
_VALID_COUNT        = 5      #lower limit of the number of tokens
_VALID_RATIO        = 10     #lower threshold for the ratio of token count to distinct token count
_PUNC_FRAC_LOW      = 0      #lower threshold for the fraction of punctuation marks
_PUNC_FRAC_HIGH     = 1/2    #upper threshold for the fraction of punctuation marks
_DUPLICATE_THRESH   = 0.5
_IRRELEVANT_THRESH  = 0.05
_TRIGGER_DAYS       = 45
_KEEP_DAYS          = 30
_TOP_NUM            = 5 
_KEYWORD_NUM        = 5
_WEIGHTS            = [1, 1, 1]
_PUNCS              = {'。', '，', '、', '：', ':', ';', '；', '“', '”', ' '}        
_SINGLES            = {'一', '二', '三', '四', '五', '六', '七', '八', '九', '十', 
                       '两', '这', '那', '不', '很', '是', '只', '就', '你', '我', 
                       '他', '她', '它', '啊', '呵', '哈', '哦'}
