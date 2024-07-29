import sys
import os
import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from scipy import stats
import gc
import argparse
import joblib as jl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from cebra import CEBRA
from hold_out import hold_out
from CSUS_score import CSUS_score
from consistency import consistency
import glob


# Adding library paths
sys.path.extend([
    '/home/hsw967/Programming/Hannahs-CEBRAs',
    '/home/hsw967/Programming/Hannahs-CEBRAs/scripts',
    '/Users/Hannah/Programming/Hannahs-CEBRAs',
    '/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra'
])


#ex
# This function measures consistency across environments for the same rat

# Global rat IDs
rat_ids = ['0222', '0307', '0313', '314', '0816']

def load_model(filename):
    # Load a model from the current working directory
    model = torch.load(filename)
    return model


def load_files(model_pattern, dimension, base_dir, divisor):
    """ Load model files based on pattern, dimension, and specific rat IDs. """
    files = []
    for rat_id in rat_ids:
        path_pattern = f"{base_dir}/rat{rat_id}/cebra_variables/models/{model_pattern}_{dimension}_*_{divisor}.pt"
        files.extend(glob.glob(path_pattern))
    return files


def delete_model_files(model_filenames):
    for filename in model_filenames:
        os.remove(filename)
        print(f"Deleted {filename}")

# Function to calculate consistency across all pairs of models
def calculate_all_models_consistency(model_data_pairs):
    """ Calculate consistency for all model-data pairs. """
    print("loading files:")
    print(filename)
    transformations = [CEBRA.load(filename).transform(data) for filename, data in model_data_pairs]
    if transformations:
        scores, pairs, ids = consistency(transformations)
        return scores, pairs, ids


def save_results(results, base_filename):
    """ Save results to a CSV file. """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{base_filename}_{current_time}.csv"
    with open(filename, 'w') as f:
        for score, pair, id in results:
            f.write(f"{score},{pair},{id}\n")
    print(f"Results saved to {filename}")


def main(args):
    base_dir = os.getcwd()
    model_patterns = ["modelAn_dim*", "modelB1_dim*", "modelAn_shuffled_dim*", "modelB1_shuffled_dim*"]
    dimensions = ["2", "3", "5", "7", "10"]
    divisor = "div2"

    for dimension in dimensions:
        for model_pattern in model_patterns:
            files = load_files(model_pattern, dimension, base_dir, divisor)
            model_data_pairs = [(file, getattr(args, f"traceR{i}A")) for i in range(1, 6) for file in files] + \
                               [(file, getattr(args, f"traceR{i}B")) for i in range(1, 6) for file in files]
            results = calculate_all_models_consistency(model_data_pairs)
            save_results(results, f"consistency_results_all_dim{dimension}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CEBRA model evaluation.")
    for i in range(1, 6):
        parser.add_argument(f"--traceR{i}A", required=True, help=f"File path for traceR{i}A data.")
        parser.add_argument(f"--traceR{i}B", required=True, help=f"File path for traceR{i}B data.")
    args = parser.parse_args()
    main(args)
