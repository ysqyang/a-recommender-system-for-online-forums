import json

def generate_recommendations(sim_matrix_path, sim_sorted_path, tid, top_num, 
                             duplicate_thresh, irrelevant_thresh):
    '''
    Given the similarity matrix, generate top_num recommendations for
    target_tid
    Args:
    sim_matrix_path:   file path for computed similarity matrix
    sim_sorted_path:   file path for sorted similarity lists
    tid:               topic id to generate recommendations fopr
    top_num:           maximum number of topics to recommend
    duplicate_thresh:  threshold value for duplicate
    irrelevant_thresh: threshold value for irrelevance 
    '''
    with open(sim_matrix_path, 'r') as f1, open(sim_sorted_path, 'r') as f2:
        sim_matrix, sim_sorted = json.load(f1), json.load(f2)

    recoms, sim_list = [], sim_sorted[tid]
    i = 0
    while len(recoms) < top_num:
        tid, sim_val = sim_list[i]
        if sim_val < irrelevant_thresh:
            break
        if sim_val < duplicate_thresh:
            if len(recoms) == 0:
                recoms.append((tid, sim_val))
            else:
                prev_tid = recoms[-1][0]
                if sim_matrix[tid][prev_tid] < duplicate_thresh:
                    recoms.append((tid, sim_val))

    return recoms