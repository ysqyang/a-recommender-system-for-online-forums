# -*- coding: utf-8 -*-
import os
#_ROOT              = '/home/ysqyang/Projects/recommender-system-for-online-forums/server'
_ROOT              = '/app/recommender/model'
#_ROOT              = '/Users/ai/Projects/recommender-system-for-online-forums'
_COMPUTED_FOLDER   = os.path.join(_ROOT, 'computed_results')
#_SERVE_LOG_FILE    = os.path.join(_LOG_FOLDER, 'serve_log')
_SIMILARITY_MATRIX = os.path.join(_COMPUTED_FOLDER, 'sim_matrix')
_SIMILARITY_SORTED = os.path.join(_COMPUTED_FOLDER, 'sim_sorted') 
_DUPLICATE_THRESH  = 0.5
