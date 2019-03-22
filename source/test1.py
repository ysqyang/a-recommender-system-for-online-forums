# -*- coding: utf-8 -*-

import pika
import json
import utils
import os, sys
import jieba
import re
from gensim.corpora import Dictionary
from utils import insert

import yaml

with open("../config/config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

for section in cfg:
    print(section)

print(cfg['preprocessing'])