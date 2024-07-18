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
import datetime

# Adding library paths
sys.path.extend([
    '/home/hsw967/Programming/Hannahs-CEBRAs',
    '/home/hsw967/Programming/Hannahs-CEBRAs/scripts',
    '/Users/Hannah/Programming/Hannahs-CEBRAs',
    '/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra'
])


#ex
##python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_saved_script4.py ./traceR1.mat ./traceR2.mat ./traceR3.mat ./traceR3.mat ./eyeblinkR1.mat ./eyeblinkAn.mat ./eyeblinkR3.mat ./eyeblinkBn.mat 2 0 --iterations 1 --parameter_set_name test

# This function measures consistency across environments for the same rat



# Function to handle the fitting and evaluation of models, and saving the top 5%
def evaluate_and_save_models(cebra_loc_model, cell_train_data, eyeblink_data, model_prefix, iterations=2):
    models = []
    losses = []
    model_data_pairs = []
    model_filenames = []

    # Train models and collect their losses
    for i in range(iterations):
        model = cebra_loc_model.fit(cell_train_data, eyeblink_data)
        loss = model.state_dict_['loss'][-1]  # Retrieve the last recorded loss
        models.append((model, loss))  # Append both model and loss

    # Sort models by loss in ascending order
    sorted_models = sorted(models, key=lambda x: x[1])

    # Determine cutoff for top 5%
    top_5_percent_index = max(1, int(len(sorted_models) * 0.05))  # Ensure at least one model is selected

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")  # Formats the datetime as Year-Month-Day_Hour-Minute-Second

    # Save only the top 5% of models
    for i in range(top_5_percent_index):
        model, loss = sorted_models[i]
        filename = f"{model_prefix}_{i}_{formatted_time}.pt"
        model.save(filename)  # Save the model
        model_filenames.append(filename)
        model_data_pairs.append((filename, cell_train_data))  # Store filename with its data

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
def save_results(results, base_filename):
    # Get the current date and time
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")  # Formats the datetime as Year-Month-Day_Hour-Minute-Second

    # Construct the filename with the current date and time
    filename = f"{base_filename}_{formatted_time}.csv"

    with open(filename, 'w') as f:
        for score, pair, id in results:
            f.write(f"{score},{pair},{id}\n")
    print(f"Results saved to {filename}"


def create_cebra_models(parameters_sets):
    cebra_models = {}
    for i, param_set in enumerate(parameters_sets, start=1):
        model_name = f"modelR{i}"  # Constructing the model name
        parameters = {
            'model_architecture': 'offset10-model',
            'batch_size': 512,
            'learning_rate': param_set["learning_rate"],
            'output_dimension': 5,
            'max_iterations': param_set["max_iterations"],
            'distance': param_set["distance"],
            'conditional': 'time_delta',
            'device': 'cuda_if_available',
            'num_hidden_units': 32,
            'time_offsets': 1,
            'verbose': False
        }
        if param_set["temp_mode"] == 'auto':
            parameters['temperature_mode'] = 'auto'
            parameters['min_temperature'] = param_set["min_temperature"]
        elif param_set["temp_mode"] == 'constant':
            parameters['temperature_mode'] = 'constant'
            parameters['temperature'] = param_set["min_temperature"]

        cebra_models[model_name] = CEBRA(**parameters)  # Store the model in the dictionary with its name as the key
        print(f"{model_name} created with temperature mode: {param_set['temp_mode']}")

    return cebra_models  #call models with cebra_models['modelR1']


def main(traceR1, traceR2, traceR3, traceR4, traceR5, trainingR1, trainingR2, trainingR3, trainingR4, trainingR5, iterations, parameter_setR1, parameter_setR2, parameter_setR3, parameter_setR4, parameter_setR5):

    # List of parameter sets
    parameter_sets = [parameter_setR1, parameter_setR2, parameter_setR3, parameter_setR4, parameter_setR5]
    cebra_models = []

    # Loop through each parameter set and create a model
    for i, param_set in enumerate(parameter_sets, start=1):
        model = create_cebra_model(param_set)
        cebra_models.append(model)
        print(f"Model R{i} created with temperature mode: {param_set['temp_mode']}")
         #call models with cebra_models['modelR1']


    # Load data from file paths provided in arguments
    traceR1_data = (traceR1)
    traceR2_data = (traceR2)
    traceR3_data = (traceR3)
    traceR4_data = (traceR4)
    traceR5_data = (traceR5)
    trainingR1_data = (trainingR1)
    trainingR2_data = (trainingR2)
    trainingR3_data = (trainingR3)
    trainingR4_data = (trainingR4)
    trainingR5_data = (trainingR5)

    envs_cell_train = [traceR1_data, traceR2_data, traceR3_data, traceR4_data, traceR5_data]
    envs_eyeblink = [trainingR1_data, trainingR2_data, trainingR3_data, trainingR4_data, trainingR5_data]

    min_length = min(len(data) for data in envs_eyeblink)
    if min_length % 10 == 9:
        envs_eyeblink = [data[9:] for data in envs_eyeblink]
        envs_cell_train = [data[9:] for data in envs_cell_train]
    # First, ensure the first 10 elements are the same
    reference_first_10 = envs_eyeblink[0][:10]  # Using the first dataset as the reference
    for i in range(1, len(envs_eyeblink)):
        while not np.array_equal(reference_first_10, envs_eyeblink[i][:10]):
            envs_eyeblink[i] = envs_eyeblink[i][1:]  # Remove the first element until the first 10 match
            envs_cell_train[i] = envs_cell_train[i][1:]
    # After aligning the first 10 elements, find the minimum length
    min_length = min(len(data) for data in envs_eyeblink)
    # Truncate all datasets to the minimum length
    envs_eyeblink = [data[:min_length] for data in envs_eyeblink]
    envs_cell_train = [data[:min_length] for data in envs_cell_train]


    ############FIX THIS
    traceA1_data = envs_cell_train[0]
    traceAn_data = envs_cell_train[1]
    traceB1_data = envs_cell_train[2]
    traceB2_data = envs_cell_train[3]
    trainingA1_data = envs_eyeblink[0]
    trainingAn_data = envs_eyeblink[1]
    trainingB1_data = envs_eyeblink[2]
    trainingB2_data = envs_eyeblink[3]
    ###########

    #call models with cebra_models['modelR1']
    # Evaluate and save models for non-shuffled data
    model_data_pairs_R1, model_filenames_R1 = evaluate_and_save_models(cebra_models['modelR1'], traceR1_data, trainingR1_data, "modelR1", iterations)
    model_data_pairs_R2, model_filenames_R2 = evaluate_and_save_models(cebra_models['modelR2'], traceR2_data, trainingR2_data, "modelR2", iterations)
    model_data_pairs_R3, model_filenames_R3 = evaluate_and_save_models(cebra_models['modelR3'], traceR3_data, trainingR3_data, "modelR3", iterations)
    model_data_pairs_R4, model_filenames_R4 = evaluate_and_save_models(cebra_models['modelR4'], traceR4_data, trainingR4_data, "modelR4", iterations)
    model_data_pairs_R5, model_filenames_R5 = evaluate_and_save_models(cebra_models['modelR5'], traceR5_data, trainingR5_data, "modelR5", iterations)


    # Evaluate and save models for shuffled data
    shuffled_index_R1 = np.random.permutation(traceR1_data.shape[0])
    cell_train_controlR1_shuffled = traceR1_data[shuffled_index_R1, :]
    model_data_pairs_R1_shuff, shuffled_filenames_R1 = evaluate_and_save_models(cebra_models['modelR1'], cell_train_controlR1_shuffled, trainingR1_data, "modelR1_shuffled", iterations)

    shuffled_index_R2 = np.random.permutation(traceR2_data.shape[0])
    cell_train_controlR2_shuffled = traceR2_data[shuffled_index_R2, :]
    model_data_pairs_R2_shuff, shuffled_filenames_R2 = evaluate_and_save_models(cebra_models['modelR2'], cell_train_controlR2_shuffled, trainingR2_data, "modelR2_shuffled", iterations)

    shuffled_index_R3 = np.random.permutation(traceR3_data.shape[0])
    cell_train_controlR3_shuffled = traceR3_data[shuffled_index_R3, :]
    model_data_pairs_R3_shuff, shuffled_filenames_R3 = evaluate_and_save_models(cebra_models['modelR3'], cell_train_controlR3_shuffled, trainingR3_data, "modelR3_shuffled", iterations)

    shuffled_index_R4 = np.random.permutation(traceR4_data.shape[0])
    cell_train_controlR4_shuffled = traceR4_data[shuffled_index_R4, :]
    model_data_pairs_R4_shuff, shuffled_filenames_R4 = evaluate_and_save_models(cebra_models['modelR4'], cell_train_controlR4_shuffled, trainingR4_data, "modelR4_shuffled", iterations)

    shuffled_index_R5 = np.random.permutation(traceR5_data.shape[0])
    cell_train_controlR5_shuffled = traceR5_data[shuffled_index_R5, :]
    model_data_pairs_R5_shuff, shuffled_filenames_R5 = evaluate_and_save_models(cebra_models['modelR5'], cell_train_controlR5_shuffled, trainingR5_data, "modelR5_shuffled", iterations)


    # Combine all pairs
    #all_model_pairs = model_data_pairs_A + model_data_pairs_B + model_data_pairs_A_shuff + model_data_pairs_B_shuff

    all_model_pairs = [
        (filename, traceR1_data) for filename in model_filenames_R1  # Non-shuffled models evaluated on shuffled data
    ] + [
        (filename, traceR2_data) for filename in model_filenames_R2  # Non-shuffled models evaluated on shuffled data
    ] + [
        (filename, traceR3_data) for filename in model_filenames_R3  # Non-shuffled models evaluated on shuffled data
    ] + [
        (filename, traceR4_data) for filename in model_filenames_R4  # Non-shuffled models evaluated on shuffled data
    ] + [
        (filename, traceR5_data) for filename in model_filenames_R5  # Non-shuffled models evaluated on shuffled data
    ] + [
        (filename, traceR1_data) for filename, _ in model_data_pairs_R1_shuff  # Shuffled models evaluated on non-shuffled data
    ] + [
        (filename, traceR2_data) for filename, _ in model_data_pairs_R2_shuff  # Shuffled models evaluated on non-shuffled data
    ] + [
        (filename, traceR3_data) for filename, _ in model_data_pairs_R3_shuff  # Shuffled models evaluated on non-shuffled data
    ] + [
        (filename, traceR4_data) for filename, _ in model_data_pairs_R4_shuff  # Shuffled models evaluated on non-shuffled data
    ] + [
        (filename, traceR5_data) for filename, _ in model_data_pairs_R5_shuff  # Shuffled models evaluated on non-shuffled data
    ]


    consistency_results_all = calculate_all_models_consistency(all_model_pairs)

    save_results(consistency_results_all, 'consistency_results_all')

    # Cleanup model files
    delete_model_files([pair[0] for pair in all_model_pairs])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CEBRA model evaluation.")
    parser.add_argument("--traceR1", required=True, help="File path for traceA data.")
    parser.add_argument("--traceR2", required=True, help="File path for traceA data.")
    parser.add_argument("--traceR3", required=True, help="File path for traceB data.")
    parser.add_argument("--traceR4", required=True, help="File path for traceB data.")
    parser.add_argument("--traceR5", required=True, help="File path for traceB data.")
    parser.add_argument("--trainingR1", required=True, help="File path for trainingA data.")
    parser.add_argument("--trainingR2", required=True, help="File path for trainingA data.")
    parser.add_argument("--trainingR3", required=True, help="File path for trainingB data.")
    parser.add_argument("--trainingR4", required=True, help="File path for trainingB data.")
    parser.add_argument("--trainingR5", required=True, help="File path for trainingB data.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations to run.")

    parser.add_argument("--learning_rateR1", type=float, default=0.01, help="Learning rate for the model.")
    parser.add_argument("--min_temperatureR1", type=float, default=0.1, help="Minimum temperature for the model.")
    parser.add_argument("--max_iterationsR1", type=int, default=100, help="Maximum iterations for the model.")
    parser.add_argument("--distanceR1", default="euclidean", help="Distance measure for the model.")
    parser.add_argument("--temp_modeR1", default="auto", help="Temperature mode for the model.")

    parser.add_argument("--learning_rateR2", type=float, default=0.01, help="Learning rate for the model.")
    parser.add_argument("--min_temperatureR2", type=float, default=0.1, help="Minimum temperature for the model.")
    parser.add_argument("--max_iterationsR2", type=int, default=100, help="Maximum iterations for the model.")
    parser.add_argument("--distanceR2", default="euclidean", help="Distance measure for the model.")
    parser.add_argument("--temp_modeR2", default="auto", help="Temperature mode for the model.")

    parser.add_argument("--learning_rateR3", type=float, default=0.01, help="Learning rate for the model.")
    parser.add_argument("--min_temperatureR3", type=float, default=0.1, help="Minimum temperature for the model.")
    parser.add_argument("--max_iterationsR3", type=int, default=100, help="Maximum iterations for the model.")
    parser.add_argument("--distanceR3", default="euclidean", help="Distance measure for the model.")
    parser.add_argument("--temp_modeR3", default="auto", help="Temperature mode for the model.")

    parser.add_argument("--learning_rateR4", type=float, default=0.01, help="Learning rate for the model.")
    parser.add_argument("--min_temperatureR4", type=float, default=0.1, help="Minimum temperature for the model.")
    parser.add_argument("--max_iterationsR4", type=int, default=100, help="Maximum iterations for the model.")
    parser.add_argument("--distanceR4", default="euclidean", help="Distance measure for the model.")
    parser.add_argument("--temp_modeR4", default="auto", help="Temperature mode for the model.")

    parser.add_argument("--learning_rateR5", type=float, default=0.01, help="Learning rate for the model.")
    parser.add_argument("--min_temperatureR5", type=float, default=0.1, help="Minimum temperature for the model.")
    parser.add_argument("--max_iterationsR5", type=int, default=100, help="Maximum iterations for the model.")
    parser.add_argument("--distanceR5", default="euclidean", help="Distance measure for the model.")
    parser.add_argument("--temp_modeR5", default="auto", help="Temperature mode for the model.")

    args = parser.parse_args()

    main(args.traceR1, args.traceR2, args.traceR3, args.traceR4, args.traceR5, args.trainingR1, args.trainingR2, args.trainingR3, args.trainingR4, args.trainingR5, args.iterations, args.parameter_setR1, args.parameter_setR2, args.parameter_setR3, args.parameter_setR4, args.parameter_setR5)
