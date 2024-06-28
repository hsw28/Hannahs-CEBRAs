import sys
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')

import numpy as np
import pandas as pd
import torch
import random
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from scipy import stats
from datetime import datetime
from cebra import CEBRA
from hold_out import hold_out
from CSUS_score import CSUS_score
import gc
import argparse
import cebra.helper as cebra_helper
from consistency import consistency
import matplotlib.pyplot as plt
import joblib as jl
from matplotlib.collections import LineCollection

#ex: python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_saved_script.py ./traceAnB1_An.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat 2 0 --iterations 2 --parameter_set_name test


# This function measures consistency across environments for the same rat
import numpy as np
import torch
from datetime import datetime
import joblib as jl


# Function to handle the fitting and evaluation of models, and saving the top 5%
def evaluate_and_save_models(cebra_loc_model, cell_train_data, eyeblink_data, model_prefix, iterations=3):
    models = []
    losses = []

    # Fit models and collect losses
    for i in range(iterations):
        model = cebra_loc_model.fit(cell_train_data, eyeblink_data)
        loss = model.state_dict_['loss'][-1]
        models.append(model)
        losses.append(loss)
        print(f"Iteration {i+1}, Loss: {loss}")

    # Sort models by their losses
    sorted_models_with_losses = sorted(zip(models, losses), key=lambda x: x[1])

    # Determine the 5% cutoff index
    cutoff_index = max(1, int(len(models) * 0.05))  # Ensure at least one model is chosen

    # Select models up to the 5% threshold, handling ties at the boundary
    selected_models = []
    last_accepted_loss = sorted_models_with_losses[cutoff_index - 1][1]
    for model, loss in sorted_models_with_losses:
        if loss <= last_accepted_loss:
            selected_models.append(model)
        if len(selected_models) >= cutoff_index:
            break

    # Save the top models within the 5% cutoff, handling ties at the boundary
    for i, model in enumerate(selected_models):
        model.save(f"{model_prefix}_{i}.pt")

    return selected_models

# Function to calculate consistency across all pairs of models
def calculate_all_pairs_consistency(models1, models2, transform1, transform2):
    results = []
    for model1 in models1:
        for model2 in models2:

            mod1results = model1.transform(transform1)
            mod2results = model2.transform(transform2)

            scores, pairs, ids = consistency([mod1results, mod2results])
            results.append((scores, pairs, ids))
    return results

# Function to save results to a CSV file
def save_results(results, filename):
    with open(filename, 'w') as f:
        for score, pair, id in results:
            f.write(f"{score},{pair},{id}\n")
    print(f"Results saved to {filename}")

# Main function to orchestrate the modeling and saving process
def main(traceA, traceB, trainingA, trainingB, iterations, parameter_set):

    learning_rate = parameter_set["learning_rate"]
    min_temperature = parameter_set["min_temperature"]
    max_iterations = parameter_set["max_iterations"]
    distance = parameter_set["distance"]
    temp_mode = parameter_set["temp_mode"]

    if temp_mode == 'auto':
        cebra_loc_model = CEBRA(model_architecture='offset10-model',
                                batch_size=512,
                                learning_rate=learning_rate,
                                temperature_mode='auto',
                                min_temperature=min_temperature,
                                output_dimension=5,
                                max_iterations=max_iterations,
                                distance=distance,
                                conditional='time_delta',
                                device='cuda_if_available',
                                num_hidden_units=32,
                                time_offsets=1,
                                verbose=True)

    elif temp_mode == 'constant':
        cebra_loc_model = CEBRA(model_architecture='offset10-model',
                                batch_size=512,
                                learning_rate=learning_rate,
                                temperature_mode='constant',
                                temperature=min_temperature,
                                output_dimension=5,
                                max_iterations=max_iterations,
                                distance=distance,
                                conditional='time_delta',
                                device='cuda_if_available',
                                num_hidden_units=32,
                                time_offsets=1,
                                verbose=True)


    # Load data from file paths provided in arguments
    envA_cell_train = (traceA)  # Adjust these as per the actual method to load your data
    envB_cell_train = (traceB)
    envA_eyeblink = (trainingA)
    envB_eyeblink = (trainingB)

    # Ensure eyeblink data is of equal length before processing
    if not np.array_equal(envA_eyeblink[:10], envB_eyeblink[:10]):
        min_length = min(len(envA_eyeblink), len(envB_eyeblink))
        envA_eyeblink, envB_eyeblink = envA_eyeblink[:min_length], envB_eyeblink[:min_length]
        envA_cell_train, envB_cell_train = envA_cell_train[:min_length], envB_cell_train[:min_length]

    # Evaluate and save models for non-shuffled data
    top_models_A = evaluate_and_save_models(cebra_loc_model, envA_cell_train, envA_eyeblink, "model1")
    top_models_B = evaluate_and_save_models(cebra_loc_model, envB_cell_train, envB_eyeblink, "model2")

    # Evaluate and save models for shuffled data
    shuffled_index_A = np.random.permutation(envA_cell_train.shape[0])
    envA_cell_train_shuffled = envA_cell_train[shuffled_index_A, :]
    top_models_A_shuff = evaluate_and_save_models(cebra_loc_model, envA_cell_train_shuffled, envA_eyeblink, "model1_shuff")

    shuffled_index_B = np.random.permutation(envB_cell_train.shape[0])
    envB_cell_train_shuffled = envB_cell_train[shuffled_index_B, :]
    top_models_B_shuff = evaluate_and_save_models(cebra_loc_model, envB_cell_train_shuffled, envB_eyeblink, "model2_shuff")

    # Calculate consistency for non-shuffled models
    consistency_results = calculate_all_pairs_consistency(top_models_A, top_models_B, envA_cell_train, envB_cell_train)
    save_results(consistency_results, "consistency_results.csv")

    # Calculate consistency for shuffled models
    consistency_results_shuff = calculate_all_pairs_consistency(top_models_A_shuff, top_models_B_shuff, envA_cell_train_shuffled, envB_cell_train_shuffled)
    save_results(consistency_results_shuff, "consistency_results_shuff.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CEBRA model evaluation.")
    parser.add_argument("--traceA", required=True, help="File path for traceA data.")
    parser.add_argument("--traceB", required=True, help="File path for traceB data.")
    parser.add_argument("--trainingA", required=True, help="File path for trainingA data.")
    parser.add_argument("--trainingB", required=True, help="File path for trainingB data.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations to run.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the model.")
    parser.add_argument("--min_temperature", type=float, default=0.1, help="Minimum temperature for the model.")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maximum iterations for the model.")
    parser.add_argument("--distance", default="euclidean", help="Distance measure for the model.")
    parser.add_argument("--temp_mode", default="auto", help="Temperature mode for the model.")
    args = parser.parse_args()

    main(args.traceA, args.traceB, args.trainingA, args.trainingB, args.iterations, args.parameter_set)
