import os
from os.path import isfile, join
from os import listdir
from itertools import permutations

import torch
import igl
import re
import numpy as np
import math
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# READ DATA IN
input_files_dir = "./data/Thingi10k";
file_names = ["output1.txt", "output2.txt", "output3.txt", "output4.txt", "output5.txt"]

data = []

for file_name in file_names:
    file_path = input_files_dir + "/" + file_name

    f = open(file_path, "r")
    lines = f.read().splitlines()
    for line in lines:
        line_data = line.split(":")
        if (len(line_data)) > 1:
            row = []
            for datum in line_data[1:]:
                row.append(float(datum))

            data.append(row)

data = np.array(data)
data_visualize = np.copy(data)

# Cap large data values
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i][j] > 1:
            # print("ij:" + str(i) + "," + str(j) + " " +str(data[i][j]))
            data_visualize[i][j] = 1
        if data[i][j] > 1e6:
            data[i][j] = 1e6


# VARIANCE
variances = data.var(axis=1)
zoomed_variances = []

#print(min(variances))
#print(max(variances))
#print(np.argmin(variances)) # 1005285.obj mesh
#print(np.argmax(variances)) # 64446.obj mesh
# Cap large variance values
for i in range(variances.shape[0]):
    if (variances[i] < 1):
        zoomed_variances.append(variances[i])

'''
# Variance histogram
print(len(zoomed_variances))
print(len(zoomed_variances) / len(variances) * 100 )
plt.hist(zoomed_variances, bins=200)
'''

# Data plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data_visualize, interpolation='nearest')
fig.colorbar(cax)
plt.show()

# AVERAGES AND MAXES
averages = np.average(data_visualize, axis=1)
maxes = np.max(data_visualize, axis=1)

#plt.hist(averages, bins=200)
#plt.hist(maxes, bins=200)
#plt.show()

# LABELLING BY AVERAGE
res = []
over_count = 0
for i in range(len(averages)):
    if averages[i] > 0.005:
        over_count += 1

#1020 over_count