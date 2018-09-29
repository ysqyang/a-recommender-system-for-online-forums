import collections
import pymysql

with open('./doc_sample.txt', 'r') as f, open('./doc_sample1.txt', 'w') as f1:
	content = f.read()
	f1.write(content.replace('\n', '')+'\n')

	













