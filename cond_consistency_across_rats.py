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
#rat_ids = ['0222', '0313', '314', '0816']
rat_ids = ['0222', '0307', '0313', '314', '0816']



def save_results(results, base_filename):
    """ Save results to a CSV file. """
    scores_runs, pairs_runs, ids_runs = results  # Unpack the three arrays

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{base_filename}_{current_time}.csv"

    with open(filename, 'w') as f:
        for score, pair in zip(scores_runs, pairs_runs):
            pair_str = '-'.join(map(str, pair))  # Convert tuple to string
            f.write(f"{pair_str},{score}\n")

    print(f"Results saved to {filename}")




def load_files(model_pattern, dimension, base_dir, divisor):
    """ Load model files based on pattern, dimension, and specific rat IDs. """
    files = []
    for rat_id in rat_ids:
        path_pattern = f"{base_dir}/rat{rat_id}/cebra_variables/models/{model_pattern}{dimension}_*_{divisor}.pt"
        matched_files = glob.glob(path_pattern)
        files.extend(matched_files)
    return files


def calculate_all_models_consistency(model_data_pairs):
    """ Calculate consistency for all model-data pairs. """
    transformations = []
    for filename, data in model_data_pairs:
        print(f"Loading model from: {filename}")
        model = CEBRA.load(filename)
        transformations.append(model.transform(data))

    if transformations:
        scores, pairs, ids = consistency(transformations)
        return scores, pairs, ids
    else:
        print("No transformations to process.")
        return None

def main(trace_data_A, trace_data_B):
    base_dir = os.getcwd()
    print(f"Using base directory: {base_dir}")
    model_patterns = ["modelAn_dim", "modelB1_dim", "modelAn_shuffled_dim", "modelB1_shuffled_dim"]
    dimensions = ["2", "3", "5", "7", "10"]
    divisor = "div2"

    for dimension in dimensions:
        all_model_data_pairs = []  # Initialize list to collect all pairs for the current dimension

        for model_pattern in model_patterns:
            files = load_files(model_pattern, dimension, base_dir, divisor)
            if files:
                for file in files:
                    # Extract rat_id from file path
                    rat_id = file.split('/rat')[1].split('/')[0]
                    index = rat_ids.index(rat_id)

                    # Select data based on model pattern
                    if "An" in model_pattern:
                        data_list = [trace_data_A[index]]  # Select corresponding A data
                    elif "B1" in model_pattern:
                        data_list = [trace_data_B[index]]  # Select corresponding B data
                    else:
                        continue  # Adjust as necessary for other patterns

                    # Append to all_model_data_pairs for the current dimension
                    all_model_data_pairs.extend([(file, data) for data in data_list])
            else:
                print(f"No files loaded for pattern {model_pattern} and dimension {dimension}.")

        # After collecting all model-data pairs for the current dimension, process them
        if all_model_data_pairs:
            results = calculate_all_models_consistency(all_model_data_pairs)
            if results:
                save_results(results, f"{dimension}_{divisor}")
            else:
                print("No results to save for dimension {}.".format(dimension))
        else:
            print(f"No model-data pairs to process for dimension {dimension}.")


    model_patterns = ["modelAn_dim", "modelB1_dim", "modelAn_shuffled_dim", "modelB1_shuffled_dim"]
    dimensions = ["2", "3", "5", "7", "10"]
    divisor = "div5"

    for dimension in dimensions:
        all_model_data_pairs = []  # Initialize list to collect all pairs for the current dimension

        for model_pattern in model_patterns:
            files = load_files(model_pattern, dimension, base_dir, divisor)
            if files:
                for file in files:
                    # Extract rat_id from file path
                    rat_id = file.split('/rat')[1].split('/')[0]
                    index = rat_ids.index(rat_id)

                    # Select data based on model pattern
                    if "An" in model_pattern:
                        data_list = [trace_data_A[index]]  # Select corresponding A data
                    elif "B1" in model_pattern:
                        data_list = [trace_data_B[index]]  # Select corresponding B data
                    else:
                        continue  # Adjust as necessary for other patterns

                    # Append to all_model_data_pairs for the current dimension
                    all_model_data_pairs.extend([(file, data) for data in data_list])
            else:
                print(f"No files loaded for pattern {model_pattern} and dimension {dimension}.")

        # After collecting all model-data pairs for the current dimension, process them
        if all_model_data_pairs:
            results = calculate_all_models_consistency(all_model_data_pairs)
            if results:
                save_results(results, f"{dimension}_{divisor}")
            else:
                print("No results to save for dimension {}.".format(dimension))
        else:
            print(f"No model-data pairs to process for dimension {dimension}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CEBRA model evaluation.")
    for i in range(1, 6):
        parser.add_argument(f"--traceR{i}A", required=True, help=f"File path for traceR{i}A data.")
        parser.add_argument(f"--traceR{i}B", required=True, help=f"File path for traceR{i}B data.")
    args = parser.parse_args()

    trace_data_A = [load_data(args.__dict__[f'traceR{i}A']) for i in range(1, 6)]
    trace_data_B = [load_data(args.__dict__[f'traceR{i}B']) for i in range(1, 6)]
    main(trace_data_A, trace_data_B)
