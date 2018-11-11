# -*- coding: utf-8 -*-

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import collections
import json
import logging
import os
import sys
sys.path.insert(0, '/home/ysqyang/Projects/recommender-system-for-online-forums/source')
#print(sys.path)
import constants as const

def serve_recommendations(request):
    '''
    Given the similarity matrix, generate top_num recommendations for
    target_tid
    '''
    logging.basicConfig(filename=const._SERVE_LOG_FILE, level=logging.DEBUG)
    if request.method == 'POST':
        return HttpResponse('METHOD not allowed!', status=405)

    print(request.GET)

    if not os.path.exists(const._SIMILARITY_MATRIX)  \
       or not os.path.exists(const._SIMILARITY_SORTED):
       return HttpResponse('No recommendations available', status=404)
        
    with open(const._SIMILARITY_MATRIX, 'r') as f1,  \
         open(const._SIMILARITY_SORTED, 'r') as f2:
        sim_matrix = json.load(f1)
        sim_sorted = json.load(f2)

    target_tid = str(request.GET['topicID'])
  
    if target_tid not in sim_sorted:
        return JsonResponse([], safe=False)

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

    return JsonResponse(recoms, safe=False)