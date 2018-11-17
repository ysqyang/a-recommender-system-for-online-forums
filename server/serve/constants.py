# -*- coding: utf-8 -*-
import os
import sys

_ROOT              = os.path.dirname(os.path.dirname(sys.path[0]))
_COMPUTED_FOLDER   = os.path.join(_ROOT, 'computed_results')
#_SERVE_LOG_FILE    = os.path.join(_LOG_FOLDER, 'serve_log')
_SIMILARITY_MATRIX = os.path.join(_COMPUTED_FOLDER, 'sim_matrix')
_SIMILARITY_SORTED = os.path.join(_COMPUTED_FOLDER, 'sim_sorted') 
_DUPLICATE_THRESH  = 0.5
_LOG_LEVEL         = 40



