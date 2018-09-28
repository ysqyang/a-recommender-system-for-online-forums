import numpy as np
import pandas as pd
import csv
import re
import collections

def build_corpus(in_file_path, out_file_path):
    '''
    Builds raw corpus from a csv file containing the original database
    Args: 
    in_file_path:  path for input file
    out_file_path: path for output file   
    Returns:
    path for raw corpus file 
    '''
    index_to_textid = {}
    d = collections.defaultdict(int)
    with open(in_file_path, 'r') as in_file, open(out_file_path, 'w') as out_file:
        cnt = 0
        reader = csv.reader(line.replace('\0', '') for line in in_file)
        writer = csv.writer(out_file)
        for line in reader:
            print(line)
            d[len(line)] += 1
            #index_to_textid[int(line[0])] = cnt
            '''
            if len(line) > 0:
            	writer.writerow(line[2])
            '''
            cnt += 1

    print(d)

    return out_file_path

in_file_path = '/home/ysqyang/Downloads/topics_info_0.csv'
out_file_path = './raw_corpus.txt'

df = pd.read_csv(in_file_path)

df.head()






