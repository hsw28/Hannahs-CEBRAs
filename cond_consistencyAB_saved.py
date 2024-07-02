import sys
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')

import numpy as np
import pandas as pd
import torch
import random
import os

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

#ex
##python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_saved_script.py ./traceAnB1_An.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat 2 0 --iterations 2 --parameter_set_name test


# This function measures consistency across environments for the same rat
import numpy as np
import torch
from datetime import datetime
import joblib as jl

import sys
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')

import numpy as np
import pandas as pd
import torch
import random
import os

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

#ex
##python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_saved_script.py ./traceAnB1_An.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat 2 0 --iterations 2 --parameter_set_name test

# This function measures consistency across environments for the same rat
import numpy as np
import torch
from datetime import datetime
import joblib as jl


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
        eyeblink_train_controlA, eyeblink_train_controlB = envA_eyeblink[:min_length], envB_eyeblink[:min_length]
        cell_train_controlA, cell_train_controlB = envA_cell_train[:min_length], envB_cell_train[:min_length]
    else:
        eyeblink_train_controlA, eyeblink_train_controlB = envA_eyeblink, envB_eyeblink
        cell_train_controlA, cell_train_controlB = envA_cell_train, envB_cell_train


    # Evaluate and save models for non-shuffled data
    model_data_pairs_A, model_filenames_A = evaluate_and_save_models(cebra_loc_model, cell_train_controlA, eyeblink_train_controlA, "modelA", iterations)
    model_data_pairs_B, model_filenames_B = evaluate_and_save_models(cebra_loc_model, cell_train_controlB, eyeblink_train_controlB, "modelB", iterations)

    # Evaluate and save models for shuffled data
    shuffled_index_A = np.random.permutation(cell_train_controlA.shape[0])
    cell_train_controlA_shuffled = cell_train_controlA[shuffled_index_A, :]
    model_data_pairs_A_shuff, shuffled_filenames_A = evaluate_and_save_models(cebra_loc_model, cell_train_controlA_shuffled, eyeblink_train_controlA, "modelA_shuffled", iterations)

    shuffled_index_B = np.random.permutation(cell_train_controlB.shape[0])
    cell_train_controlB_shuffled = cell_train_controlB[shuffled_index_B, :]
    model_data_pairs_B_shuff, shuffled_filenames_B = evaluate_and_save_models(cebra_loc_model, cell_train_controlB_shuffled, eyeblink_train_controlB, "modelB_shuffled", iterations)



    # Combine all pairs
    #all_model_pairs = model_data_pairs_A + model_data_pairs_B + model_data_pairs_A_shuff + model_data_pairs_B_shuff

    all_model_pairs = [
        (filename, cell_train_controlA) for filename in model_filenames_A  # Non-shuffled models evaluated on shuffled data
    ] + [
        (filename, cell_train_controlB) for filename in model_filenames_B  # Non-shuffled models evaluated on shuffled data
    ] + [
        (filename, cell_train_controlA) for filename, _ in model_data_pairs_A_shuff  # Shuffled models evaluated on non-shuffled data
    ] + [
        (filename, cell_train_controlB) for filename, _ in model_data_pairs_B_shuff  # Shuffled models evaluated on non-shuffled data
    ]

    consistency_results_all = calculate_all_models_consistency(all_model_pairs)
    save_results(consistency_results_all, "consistency_results_all.csv")

    # Cleanup model files
    delete_model_files([pair[0] for pair in all_model_pairs])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CEBRA model evaluation.")
    parser.add_argument("--traceA", required=True, help="File path for traceA data.")
    parser.add_argument("--traceB", required=True, help="File path for traceB data.")
    parser.add_argument("--trainingA", required=True, help="File path for trainingA data.")
    parser.add_argument("--trainingB", required=True, help="File path for trainingB data.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations to run.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the model.")
    parser.add_argument("--min_temperature", type=float, default=0.1, help="Minimum temperature for the model.")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum iterations for the model.")
    parser.add_argument("--distance", default="euclidean", help="Distance measure for the model.")
    parser.add_argument("--temp_mode", default="auto", help="Temperature mode for the model.")
    args = parser.parse_args()

    main(args.traceA, args.traceB, args.trainingA, args.trainingB, args.iterations, args.parameter_set)
