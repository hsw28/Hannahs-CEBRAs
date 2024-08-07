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
import cond_consistencyAB_saved

#ex
#python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_saved_script.py ./traceAnB1_An.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat 2 0 --iterations 2 --parameter_set_name test


# Define parameter sets
parameter_sets = parameter_sets = {
    "set0222": {"learning_rate": 0.0055, "min_temperature": 1.33, "max_iterations": 25000, "distance": 'cosine', "temp_mode": 'constant'},
    "set0307": {"learning_rate": 0.008, "min_temperature": 4, "max_iterations": 5000, "distance": 'cosine', "temp_mode": 'constant'},

    "set0307b": {"learning_rate": 0.045, "min_temperature": .84, "max_iterations": 17000, "distance": 'cosine', "temp_mode": 'auto'},
    "set0307c": {"learning_rate": 0.045, "min_temperature": .84, "max_iterations": 28000, "distance": 'cosine', "temp_mode": 'constant'},

    "set0313": {"learning_rate": 0.0035, "min_temperature": 1.67, "max_iterations": 20000, "distance": 'cosine', "temp_mode": 'auto'},
    "set0314": {"learning_rate": 0.0075, "min_temperature": 1.67, "max_iterations": 18000, "distance": 'euclidean', "temp_mode": 'constant'},
    "set0314b": {"learning_rate": 0.0075, "min_temperature": .5, "max_iterations": 16000, "distance": 'cosine', "temp_mode": 'auto'},

    "set0816": {"learning_rate": 0.0095, "min_temperature": 1.67, "max_iterations": 16000, "distance": 'cosine', "temp_mode": 'auto'},
    "set0816b": {"learning_rate": 0.00045, "min_temperature": .4, "max_iterations": 9000, "distance": 'cosine', "temp_mode": 'auto'},

    "test": {"learning_rate": 0.02, "min_temperature": .02, "max_iterations": 100, "distance": 'cosine', "temp_mode": 'auto'}
}

# Setup argparse for command line arguments
parser = argparse.ArgumentParser(description="Run decoding with CEBRA.")
parser.add_argument("traceA", type=str, help="Path to the traceAnB1_An data file.")
parser.add_argument("traceB", type=str, help="Path to the traceAnB1_B1 data file.")
parser.add_argument("trainingA", type=str, help="Path to the CSUSAn data file.")
parser.add_argument("trainingB", type=str, help="Path to the CSUSB1 data file.")
parser.add_argument("how_many_divisions", type=int, help="Number of divisions for categorizing data.")
parser.add_argument("pretrial_y_or_n", type=int, choices=[0, 1], help="Pretrial flag (0 or 1).")
parser.add_argument("--iterations", type=int, default=100, help="Number of iterations to run.")
parser.add_argument("--parameter_set_name", type=str, default="set0222", help="Name of the parameter set to use.")
args = parser.parse_args()



traceA = cebra.load_data(file=args.traceA)  # Adjust 'your_key_here' as necessary
traceB= cebra.load_data(file=args.traceB)  # Adjust 'your_key_here' as necessary
trainingA = cebra.load_data(file=args.trainingA)  # Adjust 'your_key_here' as necessary
trainingB = cebra.load_data(file=args.trainingB)  # Adjust 'your_key_here' as necessary

# Data preprocessing steps
trainingA = trainingA[0, :]
trainingB = trainingB[0, :]
traceA = np.transpose(traceA)
traceB = np.transpose(traceB)
trainingA = trainingA.flatten()
trainingB = trainingB.flatten()


# Logic to divide data based on 'divisions' and 'pretrial'
if args.pretrial_y_or_n == 0:
    traceA = traceA[trainingA > 0]
    trainingA = trainingA[trainingA > 0]

    traceB = traceB[trainingB > 0]
    trainingB = trainingB[trainingB > 0]
else:
    traceA = traceA[trainingA != 0]
    trainingA = trainingA[trainingA != 0]

    traceB = traceB[trainingB != 0]
    trainingB = trainingB[trainingB != 0]


how_many_divisions = args.how_many_divisions
if how_many_divisions == 2:
    trainingA[(trainingA > 0) & (trainingA <= 6)]  = 1
    trainingA[trainingA > 6] = 2
    trainingA[trainingA == -1] = 0

    trainingB[(trainingB > 0) & (trainingB <= 6)]  = 1
    trainingB[trainingB > 6] = 2
    trainingB[trainingB == -1] = 0
elif how_many_divisions == 5:
    trainingA[(trainingA > 0) & (trainingA <= 2)]  = 1
    trainingA[(trainingA > 2) & (trainingA <= 4)] = 2
    trainingA[(trainingA > 4) & (trainingA <= 6)] = 3
    trainingA[(trainingA > 6) & (trainingA <= 8)] = 4
    trainingA[trainingA > 8] = 5
    trainingA[trainingA == -1] = 0

    trainingB[(trainingB > 0) & (trainingB <= 2)]  = 1
    trainingB[(trainingB > 2) & (trainingB <= 4)] = 2
    trainingB[(trainingB > 4) & (trainingB <= 6)] = 3
    trainingB[(trainingB > 6) & (trainingB <= 8)] = 4
    trainingB[trainingB > 8] = 5
    trainingB[trainingB == -1] = 0


parameter_set = parameter_sets[args.parameter_set_name]
cond_consistencyAB_saved.main(traceA, traceB, trainingA, trainingB, args.iterations, parameter_set)
