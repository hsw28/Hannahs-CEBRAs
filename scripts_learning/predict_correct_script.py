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
from predict_correct import predict_correct


#ex: python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts_learning/predict_correct_script.py ./trace15_1518.mat ./trace18_1518.mat ./CSUS15.mat ./CSUS18.mat --iterations 2 --parameter_set_name try1

#ex: python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_script.py ./trace15_1518.mat ./trace18_1518.mat ./CSUS15.mat ./CSUS18.mat 2 0 --iterations 2 --parameter_set_name tr1


# Define parameter sets
parameter_sets = {
    "try1": {"learning_rate": 0.02, "min_temperature": .02, "max_iterations": 5000, "distance": 'cosine', "temp_mode": 'auto'},
    #[0.43157894736842106]
    #[0.33902439024390246]
    "try2": {"learning_rate": .002, "min_temperature": .02, "max_iterations": 5000, "distance": 'cosine', "temp_mode": 'auto'},
    #[0.5368421052631579]
    #[0.32195121951219513]
    "try3": {"learning_rate": .0002, "min_temperature": .02, "max_iterations": 5000, "distance": 'cosine', "temp_mode": 'auto'},
    #[0.5578947368421052]
    #[0.348780487804878]
    "try4": {"learning_rate": .00002, "min_temperature": .3, "max_iterations": 5000, "distance": 'cosine', "temp_mode": 'auto'},
    #[0.5578947368421052]
    #[0.375609756097561]
    "try5": {"learning_rate": .00002, "min_temperature": .5, "max_iterations": 8000, "distance": 'cosine', "temp_mode": 'auto'},
    #[0.5473684210526316]
    #[0.35853658536585364]
    "try6": {"learning_rate": .00002, "min_temperature": .3, "max_iterations": 3000, "distance": 'cosine', "temp_mode": 'auto'},
#[0.5684210526315789]
#[0.3073170731707317]

}

# Setup argparse for command line arguments
parser = argparse.ArgumentParser(description="Run decoding with CEBRA.")
parser.add_argument("traceA", type=str, help="Path to the traceAnB1_An data file.")
parser.add_argument("traceB", type=str, help="Path to the traceAnB1_B1 data file.")
parser.add_argument("trainingA", type=str, help="Path to the CSUSAn data file.")
parser.add_argument("trainingB", type=str, help="Path to the CSUSB1 data file.")


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
trainingA = trainingA.flatten()
trainingB = trainingB.flatten()

traceA = np.transpose(traceA)
traceB = np.transpose(traceB)



# Logic to divide data based on 'divisions' and 'pretrial'


parameter_set = parameter_sets[args.parameter_set_name]
#cond_consistencyAB(traceA, traceB, trainingA, trainingB, args.iterations, parameter_set)

predict_correct(traceA, traceB, trainingA, trainingB, args.iterations, parameter_set)
