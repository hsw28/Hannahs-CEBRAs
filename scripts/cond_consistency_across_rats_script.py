import sys
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')
import argparse
import gc
import numpy as np
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import joblib as jl
from itertools import product
from matplotlib.collections import LineCollection
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score
import cebra
from cebra import CEBRA
import cebra.helper as cebra_helper
from CSUS_score import CSUS_score
from hold_out import hold_out
import cond_consistency_across_rats
import warnings
import glob
import os

#ex
#python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistency_across_rats.py

# Adding library paths
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')

# Base directory is set to the current working directory
base_dir = os.getcwd()

# File path configuration
#rat_ids = ['0222', '0307', '0313', '314', '0816']
rat_ids = ['0222', '0313', '314', '0816']
trace_paths_a = [f"{base_dir}/rat{rat_id}/cebra_variables/traceAn.mat" for rat_id in rat_ids]
trace_paths_b = [f"{base_dir}/rat{rat_id}/cebra_variables/traceB1.mat" for rat_id in rat_ids]
training_paths_a = [f"{base_dir}/rat{rat_id}/cebra_variables/eyeblinkAn.mat" for rat_id in rat_ids]
training_paths_b = [f"{base_dir}/rat{rat_id}/cebra_variables/eyeblinkB1.mat" for rat_id in rat_ids]

# Data loading
trace_data_A = [cebra.load_data(file=path) for path in trace_paths_a]
trace_data_B = [cebra.load_data(file=path) for path in trace_paths_b]
training_data_a = [cebra.load_data(file=path) for path in training_paths_a]
training_data_b = [cebra.load_data(file=path) for path in training_paths_b]



# Preprocess and transpose data if necessary
for i in range(len(training_data_a)):
    training_data_a[i] = training_data_a[i][0, :].flatten()
    training_data_b[i] = training_data_b[i][0, :].flatten()
    trace_data_A[i] = np.transpose(trace_data_A[i])
    trace_data_B[i] = np.transpose(trace_data_B[i])

# Filter data based on a condition, example shown for greater than zero
for i in range(len(training_data_a)):
    mask_a = training_data_a[i] > 0
    mask_b = training_data_b[i] > 0

    training_data_a[i] = training_data_a[i][mask_a]
    trace_data_A[i] = trace_data_A[i][mask_a]

    training_data_b[i] = training_data_b[i][mask_b]
    trace_data_B[i] = trace_data_B[i][mask_b]

# Apply the specific trimming logic based on the length of the data
min_length_a = min(len(data) for data in training_data_a)
min_length_b = min(len(data) for data in training_data_b)

if min_length_a % 10 == 9:
    training_data_a = [data[9:] for data in training_data_a]
    trace_data_A = [data[9:] for data in trace_data_A]

if min_length_b % 10 == 9:
    training_data_b = [data[9:] for data in training_data_b]
    trace_data_B = [data[9:] for data in trace_data_B]

# Run the consistency analysis
cond_consistency_across_rats.main(training_data_a, training_data_b)
