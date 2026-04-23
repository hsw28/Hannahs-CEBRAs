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
from predict_correct_grid import predict_correct_grid


#ex: python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts_learning/predict_correct_grid_script.py ./trace15_1518.mat ./trace18_1518.mat ./CSUS15.mat ./CSUS18.mat --learning_rate .05,.005,.0005,.00005 --min_temperature 0.1,0.33,.67,1 --max_iterations 3333,6667,10000



def parse_list_argument(arg_value):
    """Converts a comma-separated string to a list of floats or integers."""
    try:
        return [float(item) if '.' in item else int(item) for item in arg_value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Value \"{arg_value}\" is not a valid list of numbers.")


def main():
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser(description="Run conditional decoding with CEBRA.")
    parser.add_argument("traceA", type=str, help="Path to the traceAnB1_An data file.")
    parser.add_argument("traceB", type=str, help="Path to the traceAnB1_B1 data file.")
    parser.add_argument("trainingA", type=str, help="Path to the CSUSAn data file.")
    parser.add_argument("trainingB", type=str, help="Path to the CSUSB1 data file.")

    parser.add_argument("--learning_rate", type=parse_list_argument, default=[8.6e-4], help="Comma-separated learning rates.")
    parser.add_argument("--min_temperature", type=parse_list_argument, default=[0.2], help="Comma-separated minimum temperatures.")
    parser.add_argument("--max_iterations", type=parse_list_argument, default=[8000], help="Comma-separated max iterations.")
    args = parser.parse_args()


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



    results = predict_correct_grid(
            traceA, traceB, trainingA, trainingB,
            args.learning_rate,
            args.min_temperature,
            args.max_iterations
        )

if __name__ == "__main__":
    main()
