import os
from os.path import isfile, join
from os import listdir
from itertools import permutations

import pickle
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

label_dict = {}

for file_name in file_names:
    file_path = input_files_dir + "/" + file_name

    f = open(file_path, "r")
    lines = f.read().splitlines()
    for line in lines:
        line_data = line.split(":")
        if (len(line_data)) > 1:
            row = []
            id = int(line_data[0][-8:-4])
            for datum in line_data[1:]:
                row.append(float(datum))

            avg = np.average(row)
            if avg > 0.005:
                label_dict[id] = 1
            else:
                label_dict[id] = 0

print(label_dict)
with open(input_files_dir + "/output.pkl", 'wb') as f:
    pickle.dump(label_dict, f, pickle.HIGHEST_PROTOCOL)

