from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import collections
import json
import sys
sys.path.insert(0, '/Users/ai/Projects/recommender-system-for-online-forums/source')
print(sys.path)
import constants as const

# Create your views here.
def serve_recommendations(request):
    '''
    Given the similarity matrix, generate top_num recommendations for
    target_tid
    '''
    if request.method == 'POST':
    	return HttpResponse('Posting is not allowed!')

    with open(const._SIMILARITY_MATRIX, 'r') as f:
        sim_mat = json.load(f)

    with open(const._SIMILARITY_SORTED, 'r') as f:
        sim_sorted = json.load(f)

	target_tid = request.GET['topicid'] 

    recoms = []
    for tid, sim_val in sim_sorted[target_id]: 
        if sim_val > const._DUPLICATE_THRESH:
            continue
        if sim_val < const._IRRELEVANT_THRESH:
            break
        
        if recoms == [] or sim_mat[tid][recoms[-1]] < const._DUPLICATE_THRESH:
            recoms.append(tid)
            if len(recoms) == const._TOP_NUM:
            	break

    return JsonResponse(recoms)