# -*- coding: utf-8 -*-

from django.shortcuts import render
from django.http import JsonResponse
import json
import os, sys
from datetime import datetime
print(sys.path[0])
root_dir = os.path.dirname(sys.path[0])
print('root: ', root_dir)
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

    #print(request.GET)

    try:               
        with open(const._SIMILARITY_MATRIX, 'r') as f1,  \
             open(const._SIMILARITY_SORTED, 'r') as f2:
            sim_matrix = json.load(f1)
            sim_sorted = json.load(f2)
    except Exception as e:
        logger.exception('Data file unavailable or corrupted')
        return JsonResponse({'status': True,
                             'errorCode': 2,
                             'errorMessage': 'Data file unavailable or corrupted',
                             'dto': {'list':[]},
                             '_t': datetime.now().timestamp()})

    target_tid = str(request.GET['topicID'])
  
    if target_tid not in sim_sorted:
        return JsonResponse({'status': True,
                             'errorCode': 0,
                             'errorMessage': '',
                             'dto': {'list':[]},
                             '_t': datetime.now().timestamp()})

    recoms = []
    for tid, sim_val in sim_sorted[target_tid]: 
        if sim_val > const._DUPLICATE_THRESH:
            continue
        
        if recoms == []  \
           or recoms[-1] not in sim_matrix[tid]  \
           or sim_matrix[tid][recoms[-1]] < const._DUPLICATE_THRESH:
            recoms.append(tid)
            if len(recoms) == const._TOP_NUM:
                break

    return JsonResponse({'status': True,
                         'errorCode': 0,
                         'errorMessage': '',
                         'dto': {'list':recoms},
                         '_t': datetime.now().timestamp()})