import collections
import pymysql
import numpy as np

vals = (3,6,2,5,1,7,4)
weights = [1,4,3,2]

print(np.dot(vals[2:6], weights))