_CONFIG_FILE = 'config'
_STOPWORD_FILE = 'stopwords.txt'
_DB_INFO = ('192.168.1.102','tgbweb','tgb123321','taoguba', 3307, 'utf8mb4')
_DAYS = 90
_T = 365
_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S' 
_TOPIC_FEATURES = ['TOTALVIEWNUM', 'TOTALREPLYNUM', 'POSTDATE', 
                   'USEFULNUM', 'GOLDUSEFULNUM', 'TOTALPCPOINT',
                   'TOPICPCPOINT']
_TOPIC_FILE = 'topics'
_REPLY_FILE = 'replies'
_MIN_LEN = 90
_MIN_REPLIES = 0
_MIN_REPLIES_1 = 20
_VALID_COUNT = 5
_VALID_RATIO = 2
_PUNC_FRAC_LOW = -1/20
_PUNC_FRAC_HIGH = 1/2
_IRRELEVANT_THRESH = 0.05 
_DUPLICATE_THRESH = 0.5
_TRIGGER_DAYS = 45
_KEEP_DAYS = 30
_TOPIC_ID_TO_REPLY_TABLE_NUM = './topic_id_to_reply_table_num'
_REPLY_FEATURES = ['USEFULNUM', 'GOLDUSEFULNUM', 'TOTALPCPOINT'] 
_WEIGHTS = [1, 1, 1]
_PROFILES = './profiles'
_PROFILE_WORDS = './profile_words'
_SIMILARITIES = './similarity'