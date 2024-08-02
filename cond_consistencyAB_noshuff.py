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
##python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_saved_script4.py ./traceA1.mat ./traceAn.mat ./traceB1.mat ./traceB1.mat ./eyeblinkA1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat ./eyeblinkBn.mat 2 0 --iterations 1 --parameter_set_name test

# This function measures consistency across environments for the same rat



# Function to handle the fitting and evaluation of models, and saving the top 5%
def evaluate_and_save_models(cebra_loc_model, cell_train_data, eyeblink_data, model_prefix, iterations=2):
    models = []
    losses = []
    model_data_pairs = []
    model_filenames = []

    # Train models and collect their losses
    #for i in range(iterations):
        #model = cebra_loc_model.fit(cell_train_data, eyeblink_data)
        #loss = model.state_dict_['loss'][-1]  # Retrieve the last recorded loss
        #models.append((model, loss))  # Append both model and loss

    for i in range(iterations):
        while True:  # Start an infinite loop that will keep trying until break
            model = cebra_loc_model.fit(cell_train_data, eyeblink_data)
            loss = model.state_dict_['loss'][-1]  # Retrieve the last recorded loss
            if loss < 6:
                models.append((model, loss))  # Append both model and loss
                break  # Break the infinite loop if loss is less than 6
            # If loss is not less than 6, the loop will repeat, refitting the model



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
                                output_dimension=2,
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
                                output_dimension=2,
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

    traceA1_data = envs_cell_train[0]
    traceAn_data = envs_cell_train[1]
    traceB1_data = envs_cell_train[2]
    traceB2_data = envs_cell_train[3]

    trainingA1_data = envs_eyeblink[0]
    trainingAn_data = envs_eyeblink[1]
    trainingB1_data = envs_eyeblink[2]
    trainingB2_data = envs_eyeblink[3]



    # Evaluate and save models for non-shuffled data
    model_data_pairs_A1, model_filenames_A1 = evaluate_and_save_models(cebra_loc_model, traceA1_data, trainingA1_data, "modelA1", iterations)
    model_data_pairs_An, model_filenames_An = evaluate_and_save_models(cebra_loc_model, traceAn_data, trainingAn_data, "modelAn", iterations)
    model_data_pairs_B1, model_filenames_B1 = evaluate_and_save_models(cebra_loc_model, traceB1_data, trainingB1_data, "modelB1", iterations)
    model_data_pairs_B2, model_filenames_B2 = evaluate_and_save_models(cebra_loc_model, traceB2_data, trainingB2_data, "modelB2", iterations)



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
    ]

    consistency_results_all = calculate_all_models_consistency(all_model_pairs)

    save_results(consistency_results_all, 'consistency_results_all')

    # Cleanup model files
    #delete_model_files([pair[0] for pair in all_model_pairs])

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
