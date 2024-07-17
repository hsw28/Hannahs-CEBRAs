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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime
import time

# Adding library paths
sys.path.extend([
    '/home/hsw967/Programming/Hannahs-CEBRAs',
    '/home/hsw967/Programming/Hannahs-CEBRAs/scripts',
    '/Users/Hannah/Programming/Hannahs-CEBRAs',
    '/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra'
])




#ex
##python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_saved_script4_dims.py ./traceA1.mat ./traceAn.mat ./traceB1.mat ./traceB2.mat ./eyeblinkA1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat ./eyeblinkB2.mat 2 0 --iterations 1 --parameter_set_name test

# This function measures consistency across environments for the same rat
#1 2 3 4 7 10


# Function to handle the fitting and evaluation of models, and saving the top 5%
def evaluate_and_save_models(cebra_loc_model, cell_train_data, eyeblink_data, model_prefix, iterations=20):
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
    return results


# Function to save results to a CSV file

def save_results(results, base_filename, parameter_set_name, trainingA1_data, output_dim):

    suffix = "5" if 3 in trainingA1_data else "2"
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{base_filename}_{parameter_set_name}_dim{output_dim}_{formatted_time}_div{suffix}.csv"

    # Save results to CSV
    with open(filename, 'w') as f:
        for score, pair, id in results:
            f.write(f"{score},{pair},{id}\n")
    print(f"Results saved to {filename}")


# Main function to orchestrate the modeling and saving process
def main(traceA1, traceAn, traceB1, traceB2, trainingA1, trainingAn, trainingB1, trainingB2, iterations, parameter_set, parameter_set_name):
    print(f"About to save results for parameter set: {parameter_set_name}")  # This should be a simple string like "test"

    output_dimensions = [2,3]
    for output_dim in output_dimensions:

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
                                    output_dimension=output_dim,
                                    max_iterations=max_iterations,
                                    distance=distance,
                                    conditional='time_delta',
                                    device='cuda_if_available',
                                    num_hidden_units=32,
                                    time_offsets=1,
                                    verbose=False)

        elif temp_mode == 'constant':
            cebra_loc_model = CEBRA(model_architecture='offset10-model',
                                    batch_size=512,
                                    learning_rate=learning_rate,
                                    temperature_mode='constant',
                                    temperature=min_temperature,
                                    output_dimension=output_dim,
                                    max_iterations=max_iterations,
                                    distance=distance,
                                    conditional='time_delta',
                                    device='cuda_if_available',
                                    num_hidden_units=32,
                                    time_offsets=1,
                                    verbose=False)


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

        model_data_pairs_A1, model_filenames_A1 = evaluate_and_save_models(cebra_loc_model, traceA1_data, trainingA1_data, f"modelA1_dim{output_dim}", iterations)
        model_data_pairs_An, model_filenames_An = evaluate_and_save_models(cebra_loc_model, traceAn_data, trainingAn_data, f"modelAn_dim{output_dim}", iterations)
        model_data_pairs_B1, model_filenames_B1 = evaluate_and_save_models(cebra_loc_model, traceB1_data, trainingB1_data, f"modelB1_dim{output_dim}", iterations)
        model_data_pairs_B2, model_filenames_B2 = evaluate_and_save_models(cebra_loc_model, traceB2_data, trainingB2_data, f"modelB2_dim{output_dim}", iterations)

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
        cell_train_controlB_shuffled = traceB2_data[shuffled_index_B, :]
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
            (filename, traceB2_data) for filename, _ in model_data_pairs_B2_shuff  # Shuffled models evaluated on non-shuffled data
        ]

        print(f"Saving results for parameter set: {parameter_set_name}")

        consistency_results_all = calculate_all_models_consistency(all_model_pairs)
        save_results(consistency_results_all, 'consistency_results_all', parameter_set_name, trainingA1_data, output_dim)

        #delete_model_files([pair[0] for pair in all_model_pairs])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CEBRA model evaluation.")
    parser.add_argument("--traceA1", required=True, help="File path for traceA data.")
    parser.add_argument("--traceAn", required=True, help="File path for traceAn data.")
    parser.add_argument("--traceB1", required=True, help="File path for traceB data.")
    parser.add_argument("--traceB2", required=True, help="File path for traceB data.")
    parser.add_argument("--trainingA1", required=True, help="File path for trainingA data.")
    parser.add_argument("--trainingAn", required=True, help="File path for trainingAn data.")
    parser.add_argument("--trainingB1", required=True, help="File path for trainingB data.")
    parser.add_argument("--trainingB2", required=True, help="File path for trainingB data.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations to run.")
    parser.add_argument("--parameter_set_name", required=True, help="Name of the parameter set.")

    args = parser.parse_args()

    # Access the parameter set using the name from the parameter_sets dictionary
    parameter_set = parameter_sets[args.parameter_set_name]

    # Print debugging information
    print(f"Using parameter set name: {args.parameter_set_name}")
    print(f"Using parameters: {parameter_set}")

    # Call the main function with the appropriate arguments
    main(args.traceA1, args.traceAn, args.traceB1, args.traceB2, args.trainingA1, args.trainingAn, args.trainingB1, args.trainingB2, args.iterations, parameter_set, args.parameter_set_name)
