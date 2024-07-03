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

#ex
#python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_saved_script4.py ./traceR1.mat ./traceAn.mat ./traceR3.mat ./traceR4.mat ./eyeblinkR1.mat ./eyeblinkAn.mat ./eyeblinkR3.mat ./eyeblinkR4.mat 2 0 --iterations 2 --parameter_set_name test


# Define parameter sets
parameter_sets = {
    "set0222": {"learning_rate": 0.0035, "min_temperature": 2, "max_iterations": 16000, "distance": 'cosine', "temp_mode": 'auto'},
    "set0222b": {"learning_rate": 0.0055, "min_temperature": 2, "max_iterations": 25000, "distance": 'cosine', "temp_mode": 'constant'},
    "set0222c": {"learning_rate": 0.0095, "min_temperature": 1.33, "max_iterations": 18000, "distance": 'cosine', "temp_mode": 'constant'},

    "set0307": {"learning_rate": 0.0055, "min_temperature": 1, "max_iterations": 14000, "distance": 'cosine', "temp_mode": 'constant'},
    "set0307b": {"learning_rate": 0.003, "min_temperature": 1.16, "max_iterations": 25000, "distance": 'cosine', "temp_mode": 'constant'},

    "set0313": {"learning_rate": 0.0035, "min_temperature": 1.67, "max_iterations": 20000, "distance": 'cosine', "temp_mode": 'auto'},

    "set0314": {"learning_rate": 0.0075, "min_temperature": 1.67, "max_iterations": 18000, "distance": 'euclidean', "temp_mode": 'constant'},
    "set0314b": {"learning_rate": 0.0075, "min_temperature": .5, "max_iterations": 16000, "distance": 'cosine', "temp_mode": 'auto'},

    "set0816": {"learning_rate": 0.0095, "min_temperature": 1.67, "max_iterations": 16000, "distance": 'cosine', "temp_mode": 'auto'},
    "test": {"learning_rate": 0.02, "min_temperature": .02, "max_iterations": 10, "distance": 'cosine', "temp_mode": 'auto'}
}

# Setup argparse for command line arguments
parser = argparse.ArgumentParser(description="Run decoding with CEBRA.")
parser.add_argument("traceR1", type=str, help="Path to the traceR2R3_R2 data file.")
parser.add_argument("traceR2", type=str, help="Path to the traceR2R3_R2 data file.")
parser.add_argument("traceR3", type=str, help="Path to the traceR2R3_R3 data file.")
parser.add_argument("traceR4", type=str, help="Path to the traceR2R3_R3 data file.")
parser.add_argument("traceR5", type=str, help="Path to the traceR2R3_R3 data file.")
parser.add_argument("trainingR1", type=str, help="Path to the CSUSR2 data file.")
parser.add_argument("trainingR2", type=str, help="Path to the CSUSR2 data file.")
parser.add_argument("trainingR3", type=str, help="Path to the CSUSR3 data file.")
parser.add_argument("trainingR4", type=str, help="Path to the CSUSR3 data file.")
parser.add_argument("trainingR5", type=str, help="Path to the CSUSR3 data file.")
parser.add_argument("how_many_divisions", type=int, help="Number of divisions for categorizing data.")
parser.add_argument("pretrial_y_or_n", type=int, choices=[0, 1], help="Pretrial flag (0 or 1).")
parser.add_argument("--iterations", type=int, default=100, help="Number of iterations to run.")
parser.add_argument("--parameter_set_nameR1", type=str, default="set0222", help="Name of the parameter set to use.")
parser.add_argument("--parameter_set_nameR2", type=str, default="set0222", help="Name of the parameter set to use.")
parser.add_argument("--parameter_set_nameR3", type=str, default="set0222", help="Name of the parameter set to use.")
parser.add_argument("--parameter_set_nameR4", type=str, default="set0222", help="Name of the parameter set to use.")
parser.add_argument("--parameter_set_nameR5", type=str, default="set0222", help="Name of the parameter set to use.")
args = parser.parse_args()



traceR1 = cebra.load_data(file=args.traceR1)  # Adjust 'your_key_here' as necessary
traceR2 = cebra.load_data(file=args.traceR2)  # Adjust 'your_key_here' as necessary
traceR3 = cebra.load_data(file=args.traceR3)  # Adjust 'your_key_here' as necessary
traceR4 = cebra.load_data(file=args.traceR4)  # Adjust 'your_key_here' as necessary
traceR5 = cebra.load_data(file=args.traceR5)  # Adjust 'your_key_here' as necessary

trainingR1 = cebra.load_data(file=args.trainingR1)  # Adjust 'your_key_here' as necessary
trainingR2 = cebra.load_data(file=args.trainingR2)  # Adjust 'your_key_here' as necessary
trainingR3 = cebra.load_data(file=args.trainingR3)  # Adjust 'your_key_here' as necessary
trainingR4 = cebra.load_data(file=args.trainingR4)  # Adjust 'your_key_here' as necessary
trainingR5 = cebra.load_data(file=args.trainingR5)  # Adjust 'your_key_here' as necessary



# Data preprocessing steps
trainingR1 = trainingR1[0, :]
trainingR2 = trainingR2[0, :]
trainingR3 = trainingR3[0, :]
trainingR4 = trainingR4[0, :]
trainingR5 = trainingR5[0, :]
traceR1 = np.transpose(traceR1)
traceR2 = np.transpose(traceR2)
traceR3 = np.transpose(traceR3)
traceR4 = np.transpose(traceR4)
traceR5 = np.transpose(traceR5)
trainingR1 = trainingR1.flatten()
trainingR2 = trainingR2.flatten()
trainingR3 = trainingR3.flatten()
trainingR4 = trainingR4.flatten()
trainingR5 = trainingR5.flatten()

# Logic to divide data based on 'divisions' and 'pretrial'
if args.pretrial_y_or_n == 0:
    traceR1 = traceR1[trainingR1 > 0]
    trainingR1 = trainingR1[trainingR1 > 0]

    traceR2 = traceR2[trainingR2 > 0]
    trainingR2 = trainingR2[trainingR2 > 0]

    traceR3 = traceR3[trainingR3 > 0]
    trainingR3 = trainingR3[trainingR3 > 0]

    traceR4 = traceR4[trainingR4 > 0]
    trainingR4 = trainingR4[trainingR4 > 0]

    traceR5 = traceR5[trainingR5 > 0]
    trainingR5 = trainingR5[trainingR5 > 0]

else:
    traceA = traceA[trainingA != 0]
    trainingA = trainingA[trainingA != 0]

    traceB = traceB[trainingB != 0]
    trainingB = trainingB[trainingB != 0]
    warnings.warn("havent completed code for this")



how_many_divisions = args.how_many_divisions
if how_many_divisions == 2:
    trainingR1[(trainingR1 > 0) & (trainingR1 <= 6)]  = 1
    trainingR1[trainingR1 > 6] = 2
    trainingR1[trainingR1 == -1] = 0

    trainingR2[(trainingR2 > 0) & (trainingR2 <= 6)]  = 1
    trainingR2[trainingR2 > 6] = 2
    trainingR2[trainingR2 == -1] = 0

    trainingR3[(trainingR3 > 0) & (trainingR3 <= 6)]  = 1
    trainingR3[trainingR3 > 6] = 2
    trainingR3[trainingR3 == -1] = 0

    trainingR3[(trainingR4 > 0) & (trainingR4 <= 6)]  = 1
    trainingR3[trainingR4 > 6] = 2
    trainingR3[trainingR4 == -1] = 0

    trainingR3[(trainingR5 > 0) & (trainingR5 <= 6)]  = 1
    trainingR3[trainingR5 > 6] = 2
    trainingR3[trainingR5 == -1] = 0

elif how_many_divisions == 5:
    trainingR1[(trainingR1 > 0) & (trainingR1 <= 2)]  = 1
    trainingR1[(trainingR1 > 2) & (trainingR1 <= 4)] = 2
    trainingR1[(trainingR1 > 4) & (trainingR1 <= 6)] = 3
    trainingR1[(trainingR1 > 6) & (trainingR1 <= 8)] = 4
    trainingR1[trainingR1 > 8] = 5
    trainingR1[trainingR1 == -1] = 0

    trainingR2[(trainingR2 > 0) & (trainingR2 <= 2)]  = 1
    trainingR2[(trainingR2 > 2) & (trainingR2 <= 4)] = 2
    trainingR2[(trainingR2 > 4) & (trainingR2 <= 6)] = 3
    trainingR2[(trainingR2 > 6) & (trainingR2 <= 8)] = 4
    trainingR2[trainingR2 > 8] = 5
    trainingR2[trainingR2 == -1] = 0

    trainingR3[(trainingR3 > 0) & (trainingR3 <= 2)]  = 1
    trainingR3[(trainingR3 > 2) & (trainingR3 <= 4)] = 2
    trainingR3[(trainingR3 > 4) & (trainingR3 <= 6)] = 3
    trainingR3[(trainingR3 > 6) & (trainingR3 <= 8)] = 4
    trainingR3[trainingR3 > 8] = 5
    trainingR3[trainingR3 == -1] = 0

    trainingR4[(trainingR4 > 0) & (trainingR4 <= 2)]  = 1
    trainingR4[(trainingR4 > 2) & (trainingR4 <= 4)] = 2
    trainingR4[(trainingR4 > 4) & (trainingR4 <= 6)] = 3
    trainingR4[(trainingR4 > 6) & (trainingR4 <= 8)] = 4
    trainingR4[trainingR4 > 8] = 5
    trainingR4[trainingR4 == -1] = 0

    trainingR4[(trainingR5 > 0) & (trainingR5 <= 2)]  = 1
    trainingR4[(trainingR5 > 2) & (trainingR5 <= 4)] = 2
    trainingR4[(trainingR5 > 4) & (trainingR5 <= 6)] = 3
    trainingR4[(trainingR5 > 6) & (trainingR5 <= 8)] = 4
    trainingR4[trainingR5 > 8] = 5
    trainingR4[trainingR5 == -1] = 0

parameter_setR1 = parameter_sets[args.parameter_set_nameR1]
parameter_setR2 = parameter_sets[args.parameter_set_nameR2]
parameter_setR3 = parameter_sets[args.parameter_set_nameR3]
parameter_setR4 = parameter_sets[args.parameter_set_nameR4]
parameter_setR5 = parameter_sets[args.parameter_set_nameR5]

cond_consistency_across_rats.main(traceR1, traceR2, traceR3, traceR4, traceR5, trainingR1, trainingR2, trainingR3, trainingR4, trainingR5, args.iterations, parameter_setR1, parameter_setR2, parameter_setR3, parameter_setR4, parameter_setR5)
