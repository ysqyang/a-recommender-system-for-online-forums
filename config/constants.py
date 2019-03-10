# -*- coding: utf-8 -*-
import os
import sys

ROOT               = os.path.dirname(sys.path[0])
RESULT_DIR         = os.path.join(ROOT, 'results')
DATA_DIR           = os.path.join(ROOT, 'data')
STOPWORD_FILE      = os.path.join(ROOT, 'stopwords.txt')
TOPIC_FILE         = os.path.join(DATA_DIR, 'topics')
REPLY_FILE         = os.path.join(DATA_DIR, 'replies')
#_PROFILES          = os.path.join(RESULTS_DIR, 'profiles')
#_PROFILE_WORDS     = os.path.join(RESULTS_DIR, 'profile_words')
SPECIAL_DIR        = os.path.join(RESULT_DIR, 'special_topics')
TOPIC_DIR          = os.path.join(RESULT_DIR, 'topics')
RECOM_DIR          = os.path.join(RESULT_DIR, 'recoms')
SMARTIRS_SCHEME    = 'ntn'
SLEEP_TIME         = 10
DB_INFO            = ('192.168.1.102','tgbweb','tgb123321','taoguba', 3307, 'utf8mb4')
EXCHANGE_NAME      = 'recommender'
DATETIME_FORMAT    = '%Y-%m-%d %H:%M:%S'
NUM_RESULT_DIRS    = 1000
SAVE_EVERY         = 30     # number of seconds between saves
DELETE_EVERY       = 30       # number of seconds between deletes
TIMESTAMP_FACTOR   = 1000
DAYS               = 90
TIME_DECAY_SCALE   = 30
MIN_LEN            = 90
MIN_REPLIES        = 0
MIN_REPLIES_1      = 20
VALID_COUNT        = 5      #lower limit of the number of tokens
VALID_RATIO        = 10     #lower threshold for the ratio of token count to distinct token count
PUNC_FRAC_LOW      = 0      #lower threshold for the fraction of punctuation marks
PUNC_FRAC_HIGH     = 1/2    #upper threshold for the fraction of punctuation marks
DUPLICATE_THRESH   = 0.5
IRRELEVANT_THRESH  = 0.05
TRIGGER_DAYS       = 45
KEEP_DAYS          = 30
TOP_NUM            = 5
MAX_SIZE           = 10
TOP_NUM_SPECIAL    = 10
KEYWORD_NUM        = 5
WEIGHTS            = [1, 1, 1]
PUNCS              = {'。', '，', '、', '：', ':', ';', '；', '“', '”', ' '}
SINGLES            = {'一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                       '两', '这', '那', '不', '很', '是', '只', '就', '你', '我', 
                       '他', '她', '它', '啊', '呵', '哈', '哦', '去'}
