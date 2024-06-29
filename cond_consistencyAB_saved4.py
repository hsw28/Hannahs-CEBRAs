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
# This function measures consistency across environments for the same rat
import numpy as np
import torch
from datetime import datetime
import joblib as jl

#ex:
#python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_saved_script4.py ./traceA1.mat ./traceAn.mat ./traceB1.mat ./traceB2.mat ./eyeblinkA1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat ./eyeblinkB2.mat 2 0 --iterations 2 --parameter_set_name test




def prepare_model_data_and_identifiers(cebra_loc_model, cell_train_datas, eyeblink_datas, iterations):
    all_models = []
    all_filenames = []
    identifiers = []
    transform_data = []

    for i, (cell_train, eyeblink) in enumerate(zip(cell_train_datas, eyeblink_datas)):
        model_prefix = f"model_{i}"
        top_models, model_filenames = evaluate_and_save_models(cebra_loc_model, cell_train, eyeblink, model_prefix, iterations)
        all_models.extend(top_models)
        all_filenames.extend(model_filenames)
        identifiers.extend([model_prefix] * len(top_models))
        transform_data.extend([cell_train] * len(top_models))  # Storing data for transformation

        # Shuffling
        shuffled_index = np.random.permutation(cell_train.shape[0])
        cell_train_shuffled = cell_train[shuffled_index, :]
        shuffled_prefix = f"{model_prefix}_shuff"
        top_models_shuff, model_filenames_shuff = evaluate_and_save_models(cebra_loc_model, cell_train_shuffled, eyeblink, shuffled_prefix, iterations)
        all_models.extend(top_models_shuff)
        all_filenames.extend(model_filenames_shuff)
        identifiers.extend([shuffled_prefix] * len(top_models_shuff))
        transform_data.extend([cell_train_shuffled] * len(top_models_shuff))  # Storing shuffled data for transformation

    return all_models, all_filenames, identifiers, transform_data

def evaluate_and_save_models(cebra_loc_model, cell_train_data, eyeblink_data, model_prefix, iterations=50):
    models = []
    losses = []
    model_filenames = []

    for i in range(iterations):
        model = cebra_loc_model.fit(cell_train_data, eyeblink_data)
        loss = model.state_dict_['loss'][-1]
        models.append(model)
        losses.append(loss)
        print(f"Iteration {i+1}, Loss: {loss}")

    sorted_models_with_losses = sorted(zip(models, losses), key=lambda x: x[1])
    cutoff_index = max(1, int(len(models) * 0.05))
    selected_models = [model for model, loss in sorted_models_with_losses if loss <= sorted_models_with_losses[cutoff_index - 1][1]][:cutoff_index]

    for i, model in enumerate(selected_models):
        filename = f"{model_prefix}_{i}.pt"
        model.save(filename)
        model_filenames.append(filename)

    return selected_models, model_filenames

def delete_model_files(model_filenames):
    for filename in model_filenames:
        os.remove(filename)
        print(f"Deleted {filename}")


def calculate_all_pairs_consistency(models, identifiers, data):
    results = []
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i != j:  # Ensuring not to compare the model with itself
                mod1results = model1.transform(data[i])  # Transform data using model1
                mod2results = model2.transform(data[j])  # Transform data using model2
                scores, pairs, ids = consistency([mod1results, mod2results])
                results.append({
                    "pair": f"{identifiers[i]} vs {identifiers[j]}",
                    "scores": scores,
                    "pairs": pairs,
                    "ids": ids
                })
    return results

def save_results(results, filename):
    with open(filename, 'w') as f:
        for result in results:
            score_str = ','.join(map(str, result["scores"]))
            pair_str = ','.join(map(str, result["pairs"]))
            id_str = ','.join(map(str, result["ids"]))
            f.write(f"{result['pair']},{score_str},{pair_str},{id_str}\n")
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


    cell_train_datas = [traceA1, traceAn, traceB1, traceB2]
    eyeblink_datas = [trainingA1, trainingAn, trainingB1, trainingB2]
    
    # Prepare model data and identifiers
    all_models, all_filenames, identifiers, transform_data = prepare_model_data_and_identifiers(
        cebra_loc_model, cell_train_datas, eyeblink_datas, iterations
    )

    # Calculate and save consistency results
    consistency_results = calculate_all_pairs_consistency(all_models, identifiers, transform_data)
    save_results(consistency_results, "consistency_results.csv")
    delete_model_files(all_filenames)


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
