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
from cond_compare_iterations import cond_compare_iterations
from cond_compare_iterations5 import cond_compare_iterations5


#for using with slurm to run over a bunch of iterations
# python cond_compare_iterations_script.py traceA1An_An traceAnB1_An traceA1An_A1 traceAnB1_B1 CSUSAn CSUSA1 CSUSB1 2 0 --iterations 10 --parameter_set_name set0307
# python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_compare_iterations_script.py ./traceA1An_An.mat ./traceAnB1_An.mat ./traceA1An_A1.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkA1.mat ./eyeblinkB1.mat 5 0 --iterations 11 --parameter_set_name test

# Define parameter sets
parameter_sets = {
    "set0222": {"learning_rate": 0.0035, "min_temperature": 2, "max_iterations": 16000, "distance": 'cosine', "temp_mode": 'auto'},
    "set0222b": {"learning_rate": 0.0055, "min_temperature": 2, "max_iterations": 25000, "distance": 'cosine', "temp_mode": 'constant'},
    "set0222c": {"learning_rate": 0.0095, "min_temperature": 1.33, "max_iterations": 18000, "distance": 'cosine', "temp_mode": 'constant'},
    "set0307": {"learning_rate": 0.0055, "min_temperature": 1, "max_iterations": 14000, "distance": 'cosine', "temp_mode": 'constant'},
    "set0307b": {"learning_rate": 0.003, "min_temperature": 1.16, "max_iterations": 25000, "distance": 'cosine', "temp_mode": 'constant'},
    "set0313": {"learning_rate": 0.0035, "min_temperature": 1.67, "max_iterations": 20000, "distance": 'cosine', "temp_mode": 'auto'},
    "set0314": {"learning_rate": 0.0075, "min_temperature": 1.67, "max_iterations": 18000, "distance": 'euclidean', "temp_mode": 'constant'},
    "set0816": {"learning_rate": 0.0095, "min_temperature": 1.67, "max_iterations": 16000, "distance": 'cosine', "temp_mode": 'auto'},
    "test": {"learning_rate": 0.02, "min_temperature": .02, "max_iterations": 100, "distance": 'cosine', "temp_mode": 'auto'}
}

# Setup argparse for command line arguments
parser = argparse.ArgumentParser(description="Run decoding with CEBRA.")
parser.add_argument("traceA1An_An", type=str, help="Path to the traceA1An_An data file.")
parser.add_argument("traceAnB1_An", type=str, help="Path to the traceAnB1_An data file.")
parser.add_argument("traceA1An_A1", type=str, help="Path to the traceA1An_A1 data file.")
parser.add_argument("traceAnB1_B1", type=str, help="Path to the traceAnB1_B1 data file.")
parser.add_argument("CSUSAn", type=str, help="Path to the CSUSAn data file.")
parser.add_argument("CSUSA1", type=str, help="Path to the CSUSA1 data file.")
parser.add_argument("CSUSB1", type=str, help="Path to the CSUSB1 data file.")
parser.add_argument("how_many_divisions", type=int, help="Number of divisions for categorizing data.")
parser.add_argument("pretrial_y_or_n", type=int, choices=[0, 1], help="Pretrial flag (0 or 1).")
parser.add_argument("--iterations", type=int, default=100, help="Number of iterations to run.")
parser.add_argument("--parameter_set_name", type=str, default="set0222", help="Name of the parameter set to use.")

# Parse arguments
args = parser.parse_args()

traceA1An_An = cebra.load_data(file=args.traceA1An_An)
traceAnB1_An = cebra.load_data(file=args.traceAnB1_An)
traceA1An_A1 = cebra.load_data(file=args.traceA1An_A1)
traceAnB1_B1 = cebra.load_data(file=args.traceAnB1_B1)
CSUSAn = cebra.load_data(file=args.CSUSAn)
CSUSA1 = cebra.load_data(file=args.CSUSA1)
CSUSB1 = cebra.load_data(file=args.CSUSB1)

traceA1An_An = np.transpose(traceA1An_An)
traceAnB1_An = np.transpose(traceAnB1_An)
traceA1An_A1 = np.transpose(traceA1An_A1)
traceAnB1_B1 = np.transpose(traceAnB1_B1)


CSUSAn = CSUSAn[0, :].flatten()
CSUSA1 = CSUSA1[0, :].flatten()
CSUSB1 = CSUSB1[0, :].flatten()

# Logic to divide data based on 'divisions' and 'pretrial'
if args.pretrial_y_or_n == 0:
    traceA1An_An = traceA1An_An[CSUSAn > 0]
    traceAnB1_An = traceAnB1_An[CSUSAn > 0]
    CSUSAn = CSUSAn[CSUSAn > 0]

    traceA1An_A1 = traceA1An_A1[CSUSA1 > 0]
    CSUSA1 = CSUSA1[CSUSA1 > 0]

    traceAnB1_B1 = traceAnB1_B1[CSUSB1 > 0]
    CSUSB1 = CSUSB1[CSUSB1 > 0]
else:
    traceA1An_An = traceA1An_An[CSUSAn != 0]
    traceAnB1_An = traceAnB1_An[CSUSAn != 0]
    CSUSAn = CSUSAn[CSUSAn != 0]

    traceA1An_A1 = traceA1An_A1[CSUSA1 != 0]
    CSUSA1 = CSUSA1[CSUSA1 != 0]

    traceAnB1_B1 = traceAnB1_B1[CSUSB1 != 0]
    CSUSB1 = CSUSB1[CSUSB1 != 0]

how_many_divisions = args.how_many_divisions

if how_many_divisions == 2:
    CSUSAn[(CSUSAn > 0) & (CSUSAn <= 6)] = 1
    CSUSAn[CSUSAn > 6] = 2
    CSUSAn[CSUSAn == -1] = 0

    CSUSA1[(CSUSA1 > 0) & (CSUSA1 <= 6)] = 1
    CSUSA1[CSUSA1 > 6] = 2
    CSUSA1[CSUSA1 == -1] = 0

    CSUSB1[(CSUSB1 > 0) & (CSUSB1 <= 6)] = 1
    CSUSB1[CSUSB1 > 6] = 2
    CSUSB1[CSUSB1 == -1] = 0

elif how_many_divisions == 5:
    CSUSAn[(CSUSAn > 0) & (CSUSAn <= 2)] = 1
    CSUSAn[(CSUSAn > 2) & (CSUSAn <= 4)] = 2
    CSUSAn[(CSUSAn > 4) & (CSUSAn <= 6)] = 3
    CSUSAn[(CSUSAn > 6) & (CSUSAn <= 8)] = 4
    CSUSAn[CSUSAn > 8] = 5
    CSUSAn[CSUSAn == -1] = 0

    CSUSA1[(CSUSA1 > 0) & (CSUSA1 <= 2)] = 1
    CSUSA1[(CSUSA1 > 2) & (CSUSA1 <= 4)] = 2
    CSUSA1[(CSUSA1 > 4) & (CSUSA1 <= 6)] = 3
    CSUSA1[(CSUSA1 > 6) & (CSUSA1 <= 8)] = 4
    CSUSA1[CSUSA1 > 8] = 5
    CSUSA1[CSUSA1 == -1] = 0

    CSUSB1[(CSUSB1 > 0) & (CSUSB1 <= 2)] = 1
    CSUSB1[(CSUSB1 > 2) & (CSUSB1 <= 4)] = 2
    CSUSB1[(CSUSB1 > 4) & (CSUSB1 <= 6)] = 3
    CSUSB1[(CSUSB1 > 6) & (CSUSB1 <= 8)] = 4
    CSUSB1[CSUSB1 > 8] = 5
    CSUSB1[CSUSB1 == -1] = 0

dimensions = how_many_divisions + args.pretrial_y_or_n

parameter_set = parameter_sets[args.parameter_set_name]


if how_many_divisions == 2:
    cond_compare_iterations(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, CSUSAn, CSUSA1, CSUSB1, dimensions, args.iterations, parameter_set)
elif how_many_divisions == 5:
    cond_compare_iterations5(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, CSUSAn, CSUSA1, CSUSB1, dimensions, args.iterations, parameter_set)
