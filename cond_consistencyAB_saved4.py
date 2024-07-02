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
# Adding library paths
sys.path.extend([
    '/home/hsw967/Programming/Hannahs-CEBRAs',
    '/home/hsw967/Programming/Hannahs-CEBRAs/scripts',
    '/Users/Hannah/Programming/Hannahs-CEBRAs',
    '/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra'
])


#ex
##python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_saved_script.py ./traceAnB1_An.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat 2 0 --iterations 2 --parameter_set_name test

# This function measures consistency across environments for the same rat



# Function to handle the fitting and evaluation of models, and saving the top 5%
def evaluate_and_save_models(cebra_loc_model, cell_train_data, eyeblink_data, model_prefix, iterations=2):
    models = []
    losses = []
    model_data_pairs = []
    model_filenames = []

    for i in range(iterations):
        model = cebra_loc_model.fit(cell_train_data, eyeblink_data)
        loss = model.state_dict_['loss'][-1]  # Assuming you have access to this method
        filename = f"{model_prefix}_{i}.pt"
        model.save(filename)  # Assuming `model.save()` is a valid method for CEBRA
        model_filenames.append(filename)
        model_data_pairs.append((filename, cell_train_data))

    return model_data_pairs, model_filenames




def load_model(filename):
    # Load a model from the current working directory
    model = torch.load(filename)
    return model




def delete_model_files(model_filenames):
    for filename in model_filenames:
        os.remove(filename)
        print(f"Deleted {filename}")



# Function to calculate consistency across all pairs of models
def calculate_all_models_consistency(model_data_pairs):
    loaded_models = []
    transformations = []
    results = []

    # Load all models and transform data
    for filename, data in model_data_pairs:
        model = CEBRA.load(filename)
        print(filename)
        transformations.append(model.transform(data))

    # Calculate consistency across all transformed data
    if transformations:
        scores, pairs, ids = consistency(transformations)
        results.append((scores, pairs, ids))
        print(f"Calculated consistency across all models with results: {scores}")

    return results


# Function to save results to a CSV file
def save_results(results, filename):
    with open(filename, 'w') as f:
        for score, pair, id in results:
            f.write(f"{score},{pair},{id}\n")
    print(f"Results saved to {filename}")


# Main function to orchestrate the modeling and saving process
def main(traceA1, traceAn, traceB1, traceB2, trainingA1, trainingAn, trainingB1, trainingB2, iterations, parameter_set):

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
    traceA1_data = (traceA1)
    traceAn_data = (traceAn)
    traceB1_data = (traceB1)
    traceB2_data = (traceB2)
    trainingA1_data = (trainingA1)
    trainingAn_data = (trainingAn)
    trainingB1_data = (trainingB1)
    trainingB2_data = (trainingB2)

    envs_cell_train = [traceA1_data, traceAn_data, traceB1_data, traceB2_data]
    envs_eyeblink = [trainingA1_data, trainingAn_data, trainingB1_data, trainingB2_data]

    # Ensure eyeblink data is of equal length AND CONTENT before processing
    min_length = min(min(len(eyeblink) for eyeblink in envs_eyeblink), min(len(cell) for cell in envs_cell_train))
    for i in range(4):
        if not np.array_equal(envs_eyeblink[i][:10], envs_eyeblink[(i+1) % 4][:10]):
            envs_eyeblink[i] = envs_eyeblink[i][:min_length]
            envs_cell_train[i] = envs_cell_train[i][:min_length]

    # Evaluate and save models for non-shuffled data
    model_data_pairs_A, model_filenames_A1 = evaluate_and_save_models(cebra_loc_model, traceA1_data, trainingA1_data, "modelA1", iterations)
    model_data_pairs_A, model_filenames_An = evaluate_and_save_models(cebra_loc_model, traceAn_data, trainingAn_data, "modelAn", iterations)
    model_data_pairs_B, model_filenames_B1 = evaluate_and_save_models(cebra_loc_model, traceB1_data, trainingB1_data, "modelB1", iterations)
    model_data_pairs_B, model_filenames_B2 = evaluate_and_save_models(cebra_loc_model, traceB2_data, trainingB2_data, "modelB2", iterations)

    # Evaluate and save models for shuffled data
    shuffled_index_A = np.random.permutation(traceA1_data.shape[0])
    cell_train_controlA_shuffled = traceA1_data[shuffled_index_A, :]
    model_data_pairs_A1_shuff, shuffled_filenames_A1 = evaluate_and_save_models(cebra_loc_model, cell_train_controlA_shuffled, trainingA1_data, "modelA1_shuffled", iterations)

    shuffled_index_A = np.random.permutation(traceAn_data.shape[0])
    cell_train_controlA_shuffled = traceAn_data[shuffled_index_A, :]
    model_data_pairs_An_shuff, shuffled_filenames_An = evaluate_and_save_models(cebra_loc_model, cell_train_controlA_shuffled, trainingAn_data, "modelAn_shuffled", iterations)

    shuffled_index_B = np.random.permutation(traceB1_data.shape[0])
    cell_train_controlB_shuffled = traceB1_data[shuffled_index_B, :]
    model_data_pairs_B1_shuff, shuffled_filenames_B1 = evaluate_and_save_models(cebra_loc_model, cell_train_controlB_shuffled, trainingB1_data, "modelB1_shuffled", iterations)

    shuffled_index_B = np.random.permutation(traceB2_data.shape[0])
    cell_train_controlB_shuffled = traceBn_data[shuffled_index_B, :]
    model_data_pairs_B2_shuff, shuffled_filenames_B2 = evaluate_and_save_models(cebra_loc_model, cell_train_controlB_shuffled, trainingB2_data, "modelB2_shuffled", iterations)


    # Combine all pairs
    #all_model_pairs = model_data_pairs_A + model_data_pairs_B + model_data_pairs_A_shuff + model_data_pairs_B_shuff

    all_model_pairs = [
        (filename, traceA1_data) for filename in model_filenames_A1  # Non-shuffled models evaluated on shuffled data
    ] + [
        (filename, traceAn_data) for filename in model_filenames_An  # Non-shuffled models evaluated on shuffled data
    ] + [
        (filename, traceB1_data) for filename in model_filenames_B1  # Non-shuffled models evaluated on shuffled data
    ] + [
        (filename, traceB2_data) for filename in model_filenames_B2  # Non-shuffled models evaluated on shuffled data
    ] + [
        (filename, traceA1_data) for filename, _ in model_data_pairs_A1_shuff  # Shuffled models evaluated on non-shuffled data
    ] + [
        (filename, traceAn_data) for filename, _ in model_data_pairs_An_shuff  # Shuffled models evaluated on non-shuffled data
    ] + [
        (filename, traceB1_data) for filename, _ in model_data_pairs_B1_shuff  # Shuffled models evaluated on non-shuffled data
    ] + [
        (filename, traceAn_data) for filename, _ in model_data_pairs_B2_shuff  # Shuffled models evaluated on non-shuffled data
    ]

    consistency_results_all = calculate_all_models_consistency(all_model_pairs)
    save_results(consistency_results_all, "consistency_results_all.csv")

    # Cleanup model files
    delete_model_files([pair[0] for pair in all_model_pairs])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CEBRA model evaluation.")
    parser.add_argument("--traceA1", required=True, help="File path for traceA data.")
    parser.add_argument("--traceAn", required=True, help="File path for traceA data.")
    parser.add_argument("--traceB1", required=True, help="File path for traceB data.")
    parser.add_argument("--traceB2", required=True, help="File path for traceB data.")
    parser.add_argument("--trainingA1", required=True, help="File path for trainingA data.")
    parser.add_argument("--trainingAn", required=True, help="File path for trainingA data.")
    parser.add_argument("--trainingB1", required=True, help="File path for trainingB data.")
    parser.add_argument("--trainingB2", required=True, help="File path for trainingB data.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations to run.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the model.")
    parser.add_argument("--min_temperature", type=float, default=0.1, help="Minimum temperature for the model.")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maximum iterations for the model.")
    parser.add_argument("--distance", default="euclidean", help="Distance measure for the model.")
    parser.add_argument("--temp_mode", default="auto", help="Temperature mode for the model.")
    args = parser.parse_args()

    main(args.traceA1, args.traceAn, args.traceB1, args.traceB2, args.trainingA1, args.trainingAn, args.trainingB1, args.trainingB2, args.iterations, args.parameter_set)
