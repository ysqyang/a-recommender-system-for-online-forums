import os

_ROOT = '/Users/ai/Projects/recommender-system-for-online-forums'
_COMPUTED_FOLDER = os.path.join(_ROOT, 'computed_results')
_DATA_FOLDER = os.path.join(_ROOT, 'data')
_LOG_FOLDER = os.path.join(_ROOT, 'logs')
_CONFIG_FILE = os.path.join(_ROOT, 'config')
_STOPWORD_FILE = os.path.join(_ROOT, 'stopwords.txt')
_TOPIC_FILE = os.path.join(_DATA_FOLDER, 'topics')
_REPLY_FILE = os.path.join(_DATA_FOLDER, 'replies')
_TMP = os.path.join(_DATA_FOLDER, 'topics_tmp')
_RUN_LOG_FILE = os.path.join(_LOG_FOLDER, 'run_log') 
_SERVE_LOG_FILE = os.path.join(_LOG_FOLDER, 'serve_log')
_PROFILES = os.path.join(_COMPUTED_FOLDER, 'profiles')
_PROFILE_WORDS = os.path.join(_COMPUTED_FOLDER, 'profile_words')
_SIMILARITY_MATRIX = os.path.join(_COMPUTED_FOLDER, 'sim_matrix')
_SIMILARITY_SORTED = os.path.join(_COMPUTED_FOLDER, 'sim_sorted')
_DB_INFO = ('192.168.1.102','tgbweb','tgb123321','taoguba', 3307, 'utf8mb4')
_EXCHANGE_NAME = 'recommender'
_DAYS = 90
_T = 365
_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S' 
_TOPIC_FEATURES = ['TOTALVIEWNUM', 'TOTALREPLYNUM', 'POSTDATE', 
                   'USEFULNUM', 'GOLDUSEFULNUM', 'TOTALPCPOINT',
                   'TOPICPCPOINT']
_MIN_LEN = 90
_MIN_REPLIES = 0
_MIN_REPLIES_1 = 20
_VALID_COUNT = 5
_VALID_RATIO = 2
_PUNC_FRAC_LOW = 1/20
_PUNC_FRAC_HIGH = 1/2
_IRRELEVANT_THRESH = 0.05 
_DUPLICATE_THRESH = 0.5
_TRIGGER_DAYS = 45
_KEEP_DAYS = 30
_TOP_NUM = 3
#_NUM_SIM_ENTRIES = 10
_REPLY_FEATURES = ['USEFULNUM', 'GOLDUSEFULNUM', 'TOTALPCPOINT'] 
_WEIGHTS = [1, 1, 1]