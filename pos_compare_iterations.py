import sys
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs')
from torch.nn.modules import AdaptiveMaxPool2d
from cebra import CEBRA
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from pos_score import pos_score
from plot_hippocampus3d import plot_hippocampus3d
import datetime


#does not make figures
#use to run a bunch of times
# Example usage:
# traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1 are to be defined or loaded before calling this function.
# results = pos_compare_iterations(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1)
# print(results)

def train_and_evaluate(cebra_model, trace_train, trace_test, test_trace, pos_train, pos_test, test_pos):
    cebra_model.fit(trace_train, pos_train)
    train_transformed = cebra_model.transform(trace_train)
    test_transformed = cebra_model.transform(trace_test)
    test_external_transformed = cebra_model.transform(test_trace)

    # Evaluate on held-out test data
    pos_test_score_heldout, pos_test_err_heldout, dis_mean_heldout, dis_median_heldout = pos_score(train_transformed, test_transformed, pos_train, pos_test)
    # Evaluate on external test data
    pos_test_score_external, pos_test_err_external, dis_mean_external, dis_median_external = pos_score(train_transformed, test_external_transformed, pos_train, test_pos)

    return (pos_test_score_heldout, pos_test_err_heldout, dis_mean_heldout, dis_median_heldout), (pos_test_score_external, pos_test_err_external, dis_mean_external, dis_median_external)

def pos_compare_iterations(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1, iterations):
    output_dimension = 3
    cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.001,
                        temperature_mode='auto',
                        min_temperature=0.3,
                        output_dimension=output_dimension,
                        max_iterations=8000,
                        distance='euclidean',
                        conditional='time_delta',
                        device='cuda_if_available',
                        num_hidden_units=32,
                        time_offsets=1,
                        verbose=True)

    # Prepare regular data
    traceA1An_An_train, traceA1An_An_test = hold_out(traceA1An_An, 80)
    posAn_train, posAn_test = hold_out(posAn, 80)
    traceAnB1_An_train, traceAnB1_An_test = hold_out(traceAnB1_An, 80)
    posAnB1_train, posAnB1_test = hold_out(posAn, 80)  # Assuming the same label set, adjust if different

    # Prepare shuffled data
    posAn_shuffled = np.random.permutation(posAn)
    posAn_train_shuffled, posAn_test_shuffled = hold_out(posAn_shuffled, 80)
    posAnB1_train_shuffled, posAnB1_test_shuffled = hold_out(posAn_shuffled, 80)  # Assuming the same shuffled set, adjust if different

    # Evaluate regular and shuffled datasets
    results_A1 = train_and_evaluate(cebra_model, traceA1An_An_train, traceA1An_An_test, traceA1An_A1, posAn_train, posAn_test, posA1)
    results_B1 = train_and_evaluate(cebra_model, traceAnB1_An_train, traceAnB1_An_test, traceAnB1_B1, posAnB1_train, posAnB1_test, posB1)
    results_A1_shuffled = train_and_evaluate(cebra_model, traceA1An_An_train, traceA1An_An_test, traceA1An_A1, posAn_train_shuffled, posAn_test_shuffled, posA1)
    results_B1_shuffled = train_and_evaluate(cebra_model, traceAnB1_An_train, traceAnB1_An_test, traceAnB1_B1, posAnB1_train_shuffled, posAnB1_test_shuffled, posB1)

    return {
        'A1': {
            'regular': results_A1,
            'shuffled': results_A1_shuffled
        },
        'B1': {
            'regular': results_B1,
            'shuffled': results_B1_shuffled
        }
    }

# Example usage:
# traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1 are to be defined or loaded before calling this function.
# results = pos_compare_iterations(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1)
# print(results)
This version ensures that both regular and shuffled data are used for evaluations, and it performs the evaluations on both the held-out part of the training data and the separate external test dataset for each scenario. This approach should comprehensively test the model's performance under various conditions.

User
 ok, now make it so one of the inputs is iterations (a number), and then make it so the script runs that many times and adds all your values to a big spreadsheet with a row for each iteration and columns for the four outputs of pos_score (r2, Knn_pos_err, distance_mean, distance_median).  since pos_score runs 8 times, there should be 32 columns. please make sure the columns are clearly labeled
ChatGPT
To accomplish running the model evaluations multiple times and collecting the results into a spreadsheet-like format, I'll modify the script to loop through a specified number of iterations. Each iteration will produce results from eight pos_score evaluations, and the results will be added to a structured NumPy array for easy manipulation and potentially exporting to a CSV file.

Here's the updated script with the specified modifications:

python
Copy code
import sys
import numpy as np
from cebra import CEBRA
from pos_score import pos_score

sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')

def hold_out(data, percent_to_train):
    if percent_to_train > 1:
        percent_to_train = percent_to_train / 100
    split_idx = int(len(data) * percent_to_train)
    neural_train = np.array(data[:split_idx])
    neural_test = np.array(data[split_idx:])
    return neural_train, neural_test

def train_and_evaluate(cebra_model, trace_train, trace_test, test_trace, pos_train, pos_test, test_pos):
    cebra_model.fit(trace_train, pos_train)
    train_transformed = cebra_model.transform(trace_train)
    test_transformed = cebra_model.transform(trace_test)
    test_external_transformed = cebra_model.transform(test_trace)
    return pos_score(train_transformed, test_transformed, pos_train, pos_test), pos_score(train_transformed, test_external_transformed, pos_train, test_pos)

def pos_compare_iterations(iterations, traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1):
    output_dimension = 3
    cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.001,
                        temperature_mode='auto',
                        min_temperature=0.3,
                        output_dimension=output_dimension,
                        max_iterations=8000,
                        distance='euclidean',
                        conditional='time_delta',
                        device='cuda_if_available',
                        num_hidden_units=32,
                        time_offsets=1,
                        verbose=True)

    # Create a structured array to hold the results
    dtype = [('r2', 'f8'), ('knn_pos_err', 'f8'), ('distance_mean', 'f8'), ('distance_median', 'f8')]
    results = np.zeros((iterations, 32), dtype=dtype)  # 8 evaluations per iteration, 4 outputs each

    for i in range(iterations):
        # Prepare and shuffle data
        traceA1An_An_train, traceA1An_An_test = hold_out(traceA1An_An, 80)
        posAn_train, posAn_test = hold_out(posAn, 80)
        traceAnB1_An_train, traceAnB1_An_test = hold_out(traceAnB1_An, 80)
        posAnB1_train, posAnB1_test = hold_out(posAn, 80)

        posAn_shuffled = np.random.permutation(posAn)
        posAn_train_shuffled, posAn_test_shuffled = hold_out(posAn_shuffled, 80)
        posAnB1_train_shuffled, posAnB1_test_shuffled = hold_out(posAn_shuffled, 80)

        # Regular and shuffled evaluations
        regular_A1 = train_and_evaluate(cebra_model, traceA1An_An_train, traceA1An_An_test, traceA1An_A1, posAn_train, posAn_test, posA1)
        shuffled_A1 = train_and_evaluate(cebra_model, traceA1An_An_train, traceA1An_An_test, traceA1An_A1, posAn_train_shuffled, posAn_test_shuffled, posA1)
        regular_B1 = train_and_evaluate(cebra_model, traceAnB1_An_train, traceAnB1_An_test, traceAnB1_B1, posAnB1_train, posAnB1_test, posB1)
        shuffled_B1 = train_and_evaluate(cebra_model, traceAnB1_An_train, traceAnB1_An_test, traceAnB1_B1, posAnB1_train_shuffled, posAnB1_test_shuffled, posB1)

        # Store results
        evaluations = regular_A1 + shuffled_A1 + regular_B1 + shuffled_B1
        results[i] = evaluations  # Flatten all evaluations into a single row

    return results


######need to save
