# -*- coding: utf-8 -*-

from django.shortcuts import render
from django.http import JsonResponse
import json
from collections import defaultdict
import os, sys
from datetime import datetime
root_dir = os.path.dirname(sys.path[0])
config_path = os.path.abspath(os.path.join(root_dir, 'config'))
source_path = os.path.abspath(os.path.join(root_dir, 'source'))
sys.path.insert(0, config_path)
sys.path.insert(1, source_path)
import constants as const
import log_config as lc
import utils

logger = utils.get_logger_with_config(name           = lc._SERVE_LOG_NAME, 
                                      logger_level   = lc._LOGGER_LEVEL, 
                                      handler_levels = lc._LEVELS,
                                      log_dir        = lc._LOG_DIR, 
                                      mode           = lc._MODE, 
                                      log_format     = lc._LOG_FORMAT)

def serve_recommendations(request):
    '''
    Given the similarity matrix, generate top_num recommendations for
    target_tid
    '''
    if request.method == 'POST':
        return JsonResponse({'status': True,
                             'errorCode': 1,
                             'errorMessage': 'Method not allowed!',
                             'dto': {'list':[]},
                             '_t': datetime.now().timestamp()})

    n_dirs = const._NUM_RESULT_DIRS
    result_dir = const._CORPUS_DIR

    def retrieve_data(topic_id):
        folder = str(int(topic_id) % n_dirs)
        file_name = os.path.join(result_dir, folder, topic_id)
        try:   
            with open(file_name, 'r') as f:
                sim_data = json.load(f)
            return sim_data
        except Exception as e:
            logger.exception('Data file unavailable or corrupted')

    sim_data = retrieve_data(str(request.GET['topicID']))['sim_list']
    if sim_data is None:
        return JsonResponse({'status': True,
                             'errorCode': 2,
                             'errorMessage': 'Data file unavailable or corrupted',
                             'dto': {'list':[]},
                             '_t': datetime.now().timestamp()})

    recoms = [x[0] for x in sim_data[:const._TOP_NUM]]
    return JsonResponse({'status': True,
                         'errorCode': 0,
                         'errorMessage': '',
                         'dto': {'list': recoms},
                         '_t': datetime.now().timestamp()}) 