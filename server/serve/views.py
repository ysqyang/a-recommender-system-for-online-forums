# -*- coding: utf-8 -*-

from django.shortcuts import render
from django.http import JsonResponse
import json
import logging
import os
import sys
from datetime import datetime
from . import constants as const

def serve_recommendations(request):
    '''
    Given the similarity matrix, generate top_num recommendations for
    target_tid
    '''
    logging.basicConfig(filename='log', level=const._LOG_LEVEL)
    
    if request.method == 'POST':
        logging.error('Method not allowed!')
        return JsonResponse({'status': True,
                             'errorCode': 1,
                             'errorMessage': 'Method not allowed!',
                             'dto': {'list':[]},
                             '_t': datetime.now().timestamp()})

    print(request.GET)

    try:               
        with open(const._SIMILARITY_MATRIX, 'r') as f1,  \
             open(const._SIMILARITY_SORTED, 'r') as f2:
            sim_matrix = json.load(f1)
            sim_sorted = json.load(f2)
    except:
        return JsonResponse({'status': True,
                             'errorCode': 2,
                             'errorMessage': 'Data unavailable or corrupted',
                             'dto': {'list':[]},
                             '_t': datetime.now().timestamp()})

    target_tid = str(request.GET['topicID'])
  
    if target_tid not in sim_sorted:
        logging.info('Nothing to recommend')
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

    logging.info('Found %d recommendations', len(recoms))
    return JsonResponse({'status': True,
                         'errorCode': 0,
                         'errorMessage': '',
                         'dto': {'list':recoms},
                         '_t': datetime.now().timestamp()})