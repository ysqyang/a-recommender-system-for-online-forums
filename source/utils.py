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
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger(name):
    return logging.getLogger(name)


def insert(l, id_, value, max_len):
    '''
    Helper function to insert into a list of [id, value]'s sorted
    by value, keeping the length of the list no more than max_len.
    Returns None if no insertion is performed, -1 if insertion is
    performed but no element is removed from the original list and
    the removed id if the insertion is performed and an element is
    removed from the list.
    '''
    if value == 0 or (len(l) == max_len and value < l[-1][1]):
        return

    i = 0
    while i < len(l) and l[i][1] > value:
        i += 1

    l.insert(i, [id_, value])

    if len(l) > max_len:
        deleted_id = l[-1][0]
        del l[-1]
        return deleted_id

    return ''

def remove(l, id_):
    """
    Helper function to remove from a list of [id, value]'s the entry whose
    zeroth element is -id_
    """
    if len(l) == 0:
        return

    i = 0
    while i < len(l) and l[i][0] != id_:
        i += 1

    if i == len(l):
        return

    del l[i]