# -*- coding: utf-8 -*-

import pika
import json
import utils
import os, sys
import jieba
import re
from gensim.corpora import Dictionary
from utils import insert

root_dir = os.path.dirname(sys.path[0])
config_path = os.path.abspath(os.path.join(root_dir, 'config'))
sys.path.insert(1, config_path)
import constants as const


a = 0

if a:
    print('ha')