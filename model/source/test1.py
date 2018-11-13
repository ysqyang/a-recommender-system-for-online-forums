# -*- coding: utf-8 -*-

import pika
import constants as const
import json
import configparser
import utils

with open(const._TOPIC_FILE, 'r') as f:
	topics = json.load(f)

for tid, info in topics.items():
	if '黑洞' in info['body']:
		print(tid)

