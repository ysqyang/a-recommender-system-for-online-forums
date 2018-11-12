from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import collections
import json
import logging
import os
import constants as const

def serve_recommendations(request):
    '''
    Given the similarity matrix, generate top_num recommendations for
    target_tid
    '''
    logging.basicConfig(filename=const._SERVE_LOG_FILE, level=logging.DEBUG)
    if request.method == 'POST':
        logging.error('Method not allowed!')
        return HttpResponse('Method not allowed!', status=405)

    print(request.GET)

    if not os.path.exists(const._SIMILARITY_MATRIX)  \
       or not os.path.exists(const._SIMILARITY_SORTED):
       logging.error('Data unavailable')
       return HttpResponse('Data unavailable', status=404)
        
    with open(const._SIMILARITY_MATRIX, 'r') as f1,  \
         open(const._SIMILARITY_SORTED, 'r') as f2:
        sim_matrix = json.load(f1)
        sim_sorted = json.load(f2)

    target_tid = str(request.GET['topicID'])
  
    if target_tid not in sim_sorted:
        logging.info('Nothing to recommend')
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

    logging.info('Found %d recommendations', len(recoms))
    return JsonResponse(recoms, safe=False)