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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from scipy import stats
from pos_score import pos_score
from datetime import datetime
from hold_out import hold_out


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
    return pos_score(train_transformed, test_transformed, pos_train, pos_test), pos_score(train_transformed, test_external_transformed, pos_train, test_pos)

def generate_headers():
    prefixes = ["A1An_held_out", "A1", "B1An_held_out", "B1", "SHUFF_A1An_held_out", "SHUFF_A1", "SHUFF_B1An_held_out", "SHUFF_B1"]
    metrics = ["r2", "Knn_pos_err", "distance_mean", "distance_median"]
    headers = []

    for prefix in prefixes:
        for metric in metrics:
            headers.append(f"{prefix}_{metric}")

    return ','.join(headers)

def pos_compare_iterations(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1,iterations):

    output_dimension = 3
    cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=.00000055,
                        temperature_mode='constant',
                        temperature=0.6,
                        #min_temperature=0.6,
                        output_dimension=output_dimension,
                        max_iterations=9000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        num_hidden_units=32,
                        time_offsets=1,
                        verbose=False)


    results = np.zeros((iterations, 32))  # Each iteration results in 32 outputs

    try:
        for i in range(iterations):
            traceA1An_An_train, traceA1An_An_test = hold_out(traceA1An_An, 80)
            posAn_train, posAn_test = hold_out(posAn, 80)
            traceAnB1_An_train, traceAnB1_An_test = hold_out(traceAnB1_An, 80)
            posAnB1_train, posAnB1_test = hold_out(posAn, 80)

            indices = np.random.permutation(posAn.shape[0])  # Get a permutation of the row indices
            posAn_shuffled = posAn[indices, :] #apply and shuffle

            posAn_train_shuffled, posAn_test_shuffled = hold_out(posAn_shuffled, 80)


            regular_A1 = train_and_evaluate(cebra_model, traceA1An_An_train, traceA1An_An_test, traceA1An_A1, posAn_train, posAn_test, posA1)
            shuffled_A1 = train_and_evaluate(cebra_model, traceA1An_An_train, traceA1An_An_test, traceA1An_A1, posAn_train_shuffled, posAn_test_shuffled, posA1)
            regular_B1 = train_and_evaluate(cebra_model, traceAnB1_An_train, traceAnB1_An_test, traceAnB1_B1, posAn_train, posAn_test, posB1)
            shuffled_B1 = train_and_evaluate(cebra_model, traceAnB1_An_train, traceAnB1_An_test, traceAnB1_B1, posAn_train, posAn_test_shuffled, posB1)

            # Flatten results into a single row per iteration
            results[i] = np.concatenate((np.ravel(regular_A1), np.ravel(regular_B1), np.ravel(shuffled_A1), np.ravel(shuffled_B1)))

            print(results)

        # Save the results to a CSV file with the current date and time in the filename
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"pos_compare_{current_time}.csv"
        header = generate_headers()
        np.savetxt(filename, results, delimiter=',', header=header, comments='', fmt='%.3f')  # Adjust precision as needed
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return results
