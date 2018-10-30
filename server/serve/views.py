from django.shortcuts import render
from django.http import HttpResponse
import json
import sys
sys.path.insert(0, '/Users/ai/Projects/recommender-system-for-online-forums/source')
print(sys.path)
import constants as const

# Create your views here.
def serve_recommendations(request, target_tid):
    '''
    Given the similarity matrix, generate top_num recommendations for
    target_tid
    '''
    with open(const._SIMILARITY_MATRIX, 'r') as f:
        sim_matrix = json.load(f)

    with open(const._SIMILARITY_SORTED, 'r') as f:
        sim_sorted = json.load(f)

    recoms, sim_list = [], sim_sorted[target_tid]
    i = 0
    while len(recoms) < const._TOP_NUM:
        tid, sim_val = sim_list[i]
        if sim_val < const._IRRELEVANT_THRESH:
            break
        if sim_val < const._DUPLICATE_THRESH:
            if len(recoms) == 0:
                recoms.append((tid, sim_val))
            else:
                prev_tid = recoms[-1][0]
                if sim_matrix[tid][prev_tid] < const._DUPLICATE_THRESH:
                    recoms.append((tid, sim_val))
        i += 1

    return recoms

