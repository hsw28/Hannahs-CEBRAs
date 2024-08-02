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
import cond_consistency_across_rats_all
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
rat_ids = ['0222', '0307', '0313', '314', '0816']
#rat_ids = ['0222', '0313', '314', '0816']
trace_paths_a1 = [f"{base_dir}/rat{rat_id}/cebra_variables/traceA1.mat" for rat_id in rat_ids]
trace_paths_aN = [f"{base_dir}/rat{rat_id}/cebra_variables/traceAn.mat" for rat_id in rat_ids]
trace_paths_b1 = [f"{base_dir}/rat{rat_id}/cebra_variables/traceB1.mat" for rat_id in rat_ids]
trace_paths_b2 = [f"{base_dir}/rat{rat_id}/cebra_variables/traceB2.mat" for rat_id in rat_ids]

training_paths_a1 = [f"{base_dir}/rat{rat_id}/cebra_variables/eyeblinkA1.mat" for rat_id in rat_ids]
training_paths_aN = [f"{base_dir}/rat{rat_id}/cebra_variables/eyeblinkAn.mat" for rat_id in rat_ids]
training_paths_b1 = [f"{base_dir}/rat{rat_id}/cebra_variables/eyeblinkB1.mat" for rat_id in rat_ids]
training_paths_b2 = [f"{base_dir}/rat{rat_id}/cebra_variables/eyeblinkB2.mat" for rat_id in rat_ids]


# Data loading
trace_data_A1 = [cebra.load_data(file=path) for path in trace_paths_a1]
trace_data_An = [cebra.load_data(file=path) for path in trace_paths_aN]
trace_data_B1 = [cebra.load_data(file=path) for path in trace_paths_b1]
trace_data_B2 = [cebra.load_data(file=path) for path in trace_paths_b2]
training_data_a1 = [cebra.load_data(file=path) for path in training_paths_a1]
training_data_an = [cebra.load_data(file=path) for path in training_paths_aN]
training_data_b1 = [cebra.load_data(file=path) for path in training_paths_b1]
training_data_b2 = [cebra.load_data(file=path) for path in training_paths_b2]



# Preprocess and transpose data if necessary
for i in range(len(training_data_an)):
    training_data_a1[i] = training_data_a1[i][0, :].flatten()
    training_data_an[i] = training_data_an[i][0, :].flatten()
    training_data_b1[i] = training_data_b1[i][0, :].flatten()
    training_data_b2[i] = training_data_b2[i][0, :].flatten()

    trace_data_A1[i] = np.transpose(trace_data_A1[i])
    trace_data_An[i] = np.transpose(trace_data_An[i])
    trace_data_B1[i] = np.transpose(trace_data_B1[i])
    trace_data_B2[i] = np.transpose(trace_data_B2[i])

# Filter data based on a condition, example shown for greater than zero
for i in range(len(training_data_an)):
    mask_a1 = training_data_a1[i] > 0
    mask_an = training_data_an[i] > 0
    mask_b1 = training_data_b1[i] > 0
    mask_b2 = training_data_b2[i] > 0

    training_data_a1[i] = training_data_a1[i][mask_a1]
    trace_data_A1[i] = trace_data_A1[i][mask_a1]

    training_data_an[i] = training_data_an[i][mask_an]
    trace_data_An[i] = trace_data_An[i][mask_an]

    training_data_b1[i] = training_data_b1[i][mask_b1]
    trace_data_B1[i] = trace_data_B1[i][mask_b1]

    training_data_b2[i] = training_data_b2[i][mask_b2]
    trace_data_B2[i] = trace_data_B2[i][mask_b2]


# Update training_data_an and trace_data_An based on individual conditions
for i, data in enumerate(training_data_a1):
    if len(data) % 10 == 9:
        training_data_a1[i] = data[9:]  # Modify only this specific dataset
        trace_data_A1[i] = trace_data_A1[i][9:]  # Assuming trace_data_An aligns with training_data_an

for i, data in enumerate(training_data_an):
    if len(data) % 10 == 9:
        training_data_an[i] = data[9:]  # Modify only this specific dataset
        trace_data_An[i] = trace_data_An[i][9:]  # Assuming trace_data_An aligns with training_data_an

# Update training_data_b1 and trace_data_B1 based on individual conditions
for i, data in enumerate(training_data_b1):
    if len(data) % 10 == 9:
        training_data_b1[i] = data[9:]  # Modify only this specific dataset
        trace_data_B1[i] = trace_data_B1[i][9:]  # Assuming trace_data_B1 aligns with training_data_b1

for i, data in enumerate(training_data_b2):
    if len(data) % 10 == 9:
        training_data_b2[i] = data[9:]  # Modify only this specific dataset
        trace_data_B2[i] = trace_data_B2[i][9:]  # Assuming trace_data_B1 aligns with training_data_b1


# Run the consistency analysis
cond_consistency_across_rats_all.main(trace_data_A1, trace_data_An, trace_data_B1, trace_data_B2)
