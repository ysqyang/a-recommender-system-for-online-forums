import collections
import pymysql

db = pymysql.connect(host='192.168.1.102', port=3307, 
    user='tgbweb', password='tgb123321', db='taoguba', charset='utf8mb4')

cursor = db.cursor()

n = 0
sql = 'SELECT TOPICID, BODY FROM topics_info_{}'.format(0)

cursor.execute(sql)

with open('./corpus.txt', 'w') as f:
    results = cursor.fetchall()
    cnt = 0
    for result in results:
        cnt += 1
        f.write(result[1])
        f.write('\n')
        if cnt == 5:
            break











