import csv
import re
import collections

in_file_path = '/home/ysqyang/Downloads/topics_0.csv'
out_file_path = './topics_0.csv'

regex_date_time = r'20\d\d\d\d\d\d'
regex_ip = r'\d+\.\d+\.\d+\.\d+'
regex_time = r'20\d\d-\d\d-\d\d \d\d:\d\d:\d\d'
d = collections.defaultdict(int)

with open(in_file_path, 'r') as in_csv, open(out_file_path, 'w') as out_csv:
	reader = csv.reader(line.replace('\0', '') for line in in_csv)
	writer = csv.writer(out_csv)
	cnt = cnt_abn = 0
	for line in reader:
		if len(line) != 43:
			recom_begin, recom = len(line)-3, 'NULL'
			while recom_begin >= 0 and not re.fullmatch(regex_date_time, line[recom_begin]):
				recom_begin -= 1
			
			recom_begin += 1
			recom = ','.join(line[recom_begin:len(line)-2])
			
			s1, s3 = line[:17], [recom]+line[-2:]
			
			i = 17
			inner_match_ip = re.search(regex_ip, line[17])
			inner_match_time = re.search(regex_time, line[17])
			if inner_match_ip:
				start, ip = inner_match_ip.start(), inner_match_ip.group(0) 
				s2 = [line[17][:start], ip]
				if inner_match_time:
					start, time = inner_match_time.start(), inner_match_time.group(0)
					s2 += [time] 
				s2 += line[18:recom_begin]
			else:
				while i < len(line) and not re.fullmatch(regex_ip, line[i]):
					i += 1
				ip_begin = i
				s2 = line[ip_begin:recom_begin]
				if ip_begin > 17: 
					s2 = [''.join(line[17:ip_begin])]+line[ip_begin:recom_begin]
			
			line = s1 + s2 + s3		
		writer.writerow(line)




