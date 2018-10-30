import pika
import constants as const
import json
import configparser
import utils


f = open('sim_matrix', 'r')
sim_matrix = json.load(f)
l = len(sim_matrix)
for k in sim_matrix:
    if len(sim_matrix[k]) != l:
        print(k, len(sim_matrix[k]))

print('*'*80)


f = open('sim_sorted', 'r')
sim_sorted = json.load(f)
l = len(sim_sorted)
print(l)
for k in sim_sorted:
    if len(sim_sorted[k]) != l:
        print(k, len(sim_sorted[k]))