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
import cond_consistencyAB_noshuff
import warnings

#ex
# python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_noshuff_script.py ./traceA1.mat ./traceAn.mat ./traceB1.mat ./traceB2.mat ./eyeblinkA1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat ./eyeblinkB2.mat 5 0 --iterations 3 --parameter_set_name test1

# Define parameter sets
#a is best for all
#b is optimal decoding
#c is lowest loss
parameter_sets = {
    "set0222": {"learning_rate": 0.0035, "min_temperature": 2, "max_iterations": 16000, "distance": 'cosine', "temp_mode": 'auto'},

    "set0307": {"learning_rate": 0.0055, "min_temperature": 1, "max_iterations": 14000, "distance": 'cosine', "temp_mode": 'constant'},
    "set0307b": {"learning_rate": 0.001, "min_temperature": .75, "max_iterations": 28000, "distance": 'cosine', "temp_mode": 'auto'},
    "set0307c": {"learning_rate": 0.0045, "min_temperature": .3, "max_iterations": 22000, "distance": 'cosine', "temp_mode": 'auto'},

    "set0313": {"learning_rate": 0.0035, "min_temperature": 1.67, "max_iterations": 20000, "distance": 'cosine', "temp_mode": 'auto'},
    "set0313b": {"learning_rate": 0.0075, "min_temperature": 1.67, "max_iterations": 20000, "distance": 'euclidean', "temp_mode": 'auto'},
    "set0313c": {"learning_rate": 0.0035, "min_temperature": .66, "max_iterations": 20000, "distance": 'cosine', "temp_mode": 'auto'},


    "set0314": {"learning_rate": 0.0075, "min_temperature": 1.67, "max_iterations": 18000, "distance": 'euclidean', "temp_mode": 'constant'},
    "set0314b": {"learning_rate": 0.05, "min_temperature": .66, "max_iterations": 20000, "distance": 'cosine', "temp_mode": 'auto'},
    "set0314c": {"learning_rate": 0.0075, "min_temperature": .1, "max_iterations": 25000, "distance": 'cosine', "temp_mode": 'auto'},

    "set0816": {"learning_rate": 0.0095, "min_temperature": 1.67, "max_iterations": 16000, "distance": 'cosine', "temp_mode": 'auto'},
    "set0816b": {"learning_rate": 0.0045, "min_temperature": .2, "max_iterations": 17000, "distance": 'cosine', "temp_mode": 'auto'},
    "set0816c": {"learning_rate": 0.045, "min_temperature": .2, "max_iterations": 17000, "distance": 'cosine', "temp_mode": 'auto'},


    "test": {"learning_rate": 0.02, "min_temperature": .02, "max_iterations": 10, "distance": 'cosine', "temp_mode": 'auto'},

    "test1": {"learning_rate": .00086, "min_temperature": .2, "max_iterations": 20000, "distance": 'cosine', "temp_mode": 'auto'},
    "test2": {"learning_rate": .00086, "min_temperature": .2, "max_iterations": 8000, "distance": 'cosine', "temp_mode": 'auto'}, ##ok8 ##ok3
    "test3": {"learning_rate": .000005, "min_temperature": 1, "max_iterations": 8000, "distance": 'euclidean', "temp_mode": 'auto'}, ##XXX8 XXX3
    "test4": {"learning_rate": .000005, "min_temperature": .74, "max_iterations": 8000, "distance": 'euclidean', "temp_mode": 'auto'}, ##XXX8 XXX3
    "test5": {"learning_rate": .001, "min_temperature": 1, "max_iterations": 15000, "distance": 'cosine', "temp_mode": 'auto'}, ##ok8

    "test6": {"learning_rate": .00086, "min_temperature": .2, "max_iterations": 15000, "distance": 'cosine', "temp_mode": 'auto'}, ###******
    "test7": {"learning_rate": .000005, "min_temperature": .74, "max_iterations": 5000, "distance": 'euclidean', "temp_mode": 'auto'}, ##XXX3

    "test8": {"learning_rate": .00086, "min_temperature": .5, "max_iterations": 5500, "distance": 'euclidean', "temp_mode": 'auto'}, ##mehok3
    "test9": {"learning_rate": .00086, "min_temperature": .5, "max_iterations": 7000, "distance": 'euclidean', "temp_mode": 'auto'},
    "test10": {"learning_rate": .00086, "min_temperature": .5, "max_iterations": 7000, "distance": 'euclidean', "temp_mode": 'constant'},
    "test11": {"learning_rate": .00086, "min_temperature": .7, "max_iterations": 7200, "distance": 'euclidean', "temp_mode": 'constant'},


}

# Setup argparse for command line arguments
parser = argparse.ArgumentParser(description="Run decoding with CEBRA.")
parser.add_argument("traceA1", type=str, help="Path to the traceAnB1_An data file.")
parser.add_argument("traceAn", type=str, help="Path to the traceAnB1_An data file.")
parser.add_argument("traceB1", type=str, help="Path to the traceAnB1_B1 data file.")
parser.add_argument("traceB2", type=str, help="Path to the traceAnB1_B1 data file.")
parser.add_argument("trainingA1", type=str, help="Path to the CSUSAn data file.")
parser.add_argument("trainingAn", type=str, help="Path to the CSUSAn data file.")
parser.add_argument("trainingB1", type=str, help="Path to the CSUSB1 data file.")
parser.add_argument("trainingB2", type=str, help="Path to the CSUSB1 data file.")
parser.add_argument("how_many_divisions", type=int, help="Number of divisions for categorizing data.")
parser.add_argument("pretrial_y_or_n", type=int, choices=[0, 1], help="Pretrial flag (0 or 1).")
parser.add_argument("--iterations", type=int, default=100, help="Number of iterations to run.")
parser.add_argument("--parameter_set_name", type=str, default="set0222", help="Name of the parameter set to use.")
args = parser.parse_args()



traceA1 = cebra.load_data(file=args.traceA1)  # Adjust 'your_key_here' as necessary
traceAn = cebra.load_data(file=args.traceAn)  # Adjust 'your_key_here' as necessary

traceB1 = cebra.load_data(file=args.traceB1)  # Adjust 'your_key_here' as necessary
traceB2 = cebra.load_data(file=args.traceB2)  # Adjust 'your_key_here' as necessary

trainingA1 = cebra.load_data(file=args.trainingA1)  # Adjust 'your_key_here' as necessary
trainingAn = cebra.load_data(file=args.trainingAn)  # Adjust 'your_key_here' as necessary

trainingB1 = cebra.load_data(file=args.trainingB1)  # Adjust 'your_key_here' as necessary
trainingB2 = cebra.load_data(file=args.trainingB2)  # Adjust 'your_key_here' as necessary


# Data preprocessing steps
trainingA1 = trainingA1[0, :]
trainingAn = trainingAn[0, :]
trainingB1 = trainingB1[0, :]
trainingB2 = trainingB2[0, :]
traceA1 = np.transpose(traceA1)
traceAn = np.transpose(traceAn)
traceB1 = np.transpose(traceB1)
traceB2 = np.transpose(traceB2)
trainingA1 = trainingA1.flatten()
trainingAn = trainingAn.flatten()
trainingB1 = trainingB1.flatten()
trainingB2 = trainingB2.flatten()

# Logic to divide data based on 'divisions' and 'pretrial'
if args.pretrial_y_or_n == 0:
    traceA1 = traceA1[trainingA1 > 0]
    trainingA1 = trainingA1[trainingA1 > 0]

    traceAn = traceAn[trainingAn > 0]
    trainingAn = trainingAn[trainingAn > 0]

    traceB1 = traceB1[trainingB1 > 0]
    trainingB1 = trainingB1[trainingB1 > 0]

    traceB2 = traceB2[trainingB2 > 0]
    trainingB2 = trainingB2[trainingB2 > 0]
else:
    traceA = traceA[trainingA != 0]
    trainingA = trainingA[trainingA != 0]

    traceB = traceB[trainingB != 0]
    trainingB = trainingB[trainingB != 0]
    warnings.warn("havent completed code for this")



how_many_divisions = args.how_many_divisions
if how_many_divisions == 2:
    trainingA1[(trainingA1 > 0) & (trainingA1 <= 6)]  = 1
    trainingA1[trainingA1 > 6] = 2
    trainingA1[trainingA1 == -1] = 0

    trainingAn[(trainingAn > 0) & (trainingAn <= 6)]  = 1
    trainingAn[trainingAn > 6] = 2
    trainingAn[trainingAn == -1] = 0

    trainingB1[(trainingB1 > 0) & (trainingB1 <= 6)]  = 1
    trainingB1[trainingB1 > 6] = 2
    trainingB1[trainingB1 == -1] = 0

    trainingB2[(trainingB2 > 0) & (trainingB2 <= 6)]  = 1
    trainingB2[trainingB2 > 6] = 2
    trainingB2[trainingB2 == -1] = 0

elif how_many_divisions == 5:
    trainingA1[(trainingA1 > 0) & (trainingA1 <= 2)]  = 1
    trainingA1[(trainingA1 > 2) & (trainingA1 <= 4)] = 2
    trainingA1[(trainingA1 > 4) & (trainingA1 <= 6)] = 3
    trainingA1[(trainingA1 > 6) & (trainingA1 <= 8)] = 4
    trainingA1[trainingA1 > 8] = 5
    trainingA1[trainingA1 == -1] = 0

    trainingAn[(trainingAn > 0) & (trainingAn <= 2)]  = 1
    trainingAn[(trainingAn > 2) & (trainingAn <= 4)] = 2
    trainingAn[(trainingAn > 4) & (trainingAn <= 6)] = 3
    trainingAn[(trainingAn > 6) & (trainingAn <= 8)] = 4
    trainingAn[trainingAn > 8] = 5
    trainingAn[trainingAn == -1] = 0

    trainingB1[(trainingB1 > 0) & (trainingB1 <= 2)]  = 1
    trainingB1[(trainingB1 > 2) & (trainingB1 <= 4)] = 2
    trainingB1[(trainingB1 > 4) & (trainingB1 <= 6)] = 3
    trainingB1[(trainingB1 > 6) & (trainingB1 <= 8)] = 4
    trainingB1[trainingB1 > 8] = 5
    trainingB1[trainingB1 == -1] = 0


    trainingB2[(trainingB2 > 0) & (trainingB2 <= 2)]  = 1
    trainingB2[(trainingB2 > 2) & (trainingB2 <= 4)] = 2
    trainingB2[(trainingB2 > 4) & (trainingB2 <= 6)] = 3
    trainingB2[(trainingB2 > 6) & (trainingB2 <= 8)] = 4
    trainingB2[trainingB2 > 8] = 5
    trainingB2[trainingB2 == -1] = 0


parameter_set = parameter_sets[args.parameter_set_name]
cond_consistencyAB_noshuff.main(traceA1, traceAn, traceB1, traceB2, trainingA1, trainingAn, trainingB1, trainingB2, args.iterations, parameter_set)
