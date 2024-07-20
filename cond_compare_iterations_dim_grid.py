import sys
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs')
from itertools import product
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

#decodes conditioning in envB using envA.
#Outputs percent correct in envA after being trained in env A(based on a 75/25 split)
#Outputs percent correct in envB using the model trained in envA
#does not make figures
#use to run a bunch of times
# Example usage:
# traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, CSUSAn, CSUSA1, CSUSB1 are to be defined or loaded before calling this function.
# results = pos_compare_iterations(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, CSUSAn, CSUSA1, CSUSB1)
# print(results)
#     results = pos_compare_iterations(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, CSUSAn, CSUSA1, CSUSB1, args.iterations, args.parameter_set_name)



def train_and_evaluate(cebra_model, trace_train, trace_test, test_trace, pos_train, pos_test, test_pos):
    cebra_model.fit(trace_train, pos_train)
    train_transformed = cebra_model.transform(trace_train)
    test_transformed = cebra_model.transform(trace_test)
    test_external_transformed = cebra_model.transform(test_trace)
    return CSUS_score(train_transformed, test_transformed, pos_train, pos_test), CSUS_score(train_transformed, test_external_transformed, pos_train, test_pos)


def generate_headers():
    prefixes = ["A1An_held_out", "A1", "B1An_held_out", "B1", "SHUFF_A1An_held_out", "SHUFF_A1", "SHUFF_B1An_held_out", "SHUFF_B1", "output_dimension"]
    metrics = ["% correct"]
    headers = []

    for prefix in prefixes:
        for metric in metrics:
            headers.append(f"{prefix}_{metric}")

    return ','.join(headers)

def cond_compare_iterations_dim_grid(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, CSUSAn, CSUSA1, CSUSB1, iterations, parameter_set, dimensions):

    all_results = []
    learning_rate = parameter_set["learning_rate"]
    min_temperature = parameter_set["min_temperature"]
    max_iterations = parameter_set["max_iterations"]
    distance = parameter_set["distance"]
    temp_mode = parameter_set["temp_mode"]

    for dm in dimensions:

        if temp_mode == 'auto':
            cebra_model = CEBRA(
                                learning_rate=learning_rate,
                                max_iterations=max_iterations,
                                model_architecture='offset10-model',
                                batch_size=512,
                                temperature_mode='auto',
                                min_temperature=min_temperature,
                                output_dimension=dm,
                                distance=distance,
                                conditional='time_delta',
                                device='cuda_if_available',
                                num_hidden_units=32,
                                time_offsets=1,
                                verbose=False)

        if temp_mode == 'constant':
                cebra_model = CEBRA(
                                learning_rate=learning_rate,
                                max_iterations=max_iterations,
                                model_architecture='offset10-model',
                                batch_size=512,
                                temperature_mode='constant',
                                temperature=min_temperature,
                                output_dimension=dm,
                                distance=distance,
                                conditional='time_delta',
                                device='cuda_if_available',
                                num_hidden_units=32,
                                time_offsets=1,
                                verbose=False)

        results = np.zeros((iterations, 9))  # Each iteration results in 8 outputs

        min_length = (len(CSUSAn))
        if min_length % 10 == 9:
            CSUSAn = [CSUSAn[9:] for data in envs_eyeblink]
            traceA1An_An = [traceA1An_An[9:] for data in envs_cell_train]
            traceAnB1_An = [traceAnB1_An[9:] for data in envs_cell_train]

        min_length = (len(CSUSB1))
        if min_length % 10 == 9:
            CSUSB1 = [CSUSB1[9:] for data in envs_eyeblink]
            traceAnB1_B1 = [traceAnB1_B1[9:] for data in envs_cell_train]


        for i in range(iterations):

            traceA1An_An_train, traceA1An_An_test = hold_out(traceA1An_An, 75)
            CSUSAn_train, CSUSAn_test = hold_out(CSUSAn, 75)
            traceAnB1_An_train, traceAnB1_An_test = hold_out(traceAnB1_An, 75)
            CSUSAnB1_train, CSUSAnB1_test = hold_out(CSUSAn, 75)

            indices = np.random.permutation(CSUSAn.shape[0])

            CSUSAn_shuffled = CSUSAn[indices]
            CSUSAn_train_shuffled, CSUSAn_test_shuffled = hold_out(CSUSAn_shuffled, 75)


            regular_A1 = train_and_evaluate(cebra_model, traceA1An_An_train, traceA1An_An_test, traceA1An_A1, CSUSAn_train, CSUSAn_test, CSUSA1)
            regular_B1 = train_and_evaluate(cebra_model, traceAnB1_An_train, traceAnB1_An_test, traceAnB1_B1, CSUSAn_train, CSUSAn_test, CSUSB1)
            shuffled_A1 = train_and_evaluate(cebra_model, traceA1An_An_train, traceA1An_An_test, traceA1An_A1, CSUSAn_train_shuffled, CSUSAn_test_shuffled, CSUSA1)
            shuffled_B1 = train_and_evaluate(cebra_model, traceAnB1_An_train, traceAnB1_An_test, traceAnB1_B1, CSUSAn_train_shuffled, CSUSAn_test_shuffled, CSUSB1)

            # Combine results into a single row
            result_row = np.concatenate((np.ravel(regular_A1), np.ravel(regular_B1), np.ravel(shuffled_A1), np.ravel(shuffled_B1), [dm]))
            all_results.append(result_row)  # Append the row to the all_results list



    # Outside the loop: save all results to a CSV file
    all_results_array = np.array(all_results)
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"cond_lr{learning_rate}_mt{min_temperature}_mi{max_iterations}_d{distance}_mode{temp_mode}_{current_time}_DIM_GRID.csv"
    header = generate_headers()
    np.savetxt(filename, all_results_array, delimiter=',', header=header, comments='', fmt='%.3f')  # Adjust precision as needed
    print(f"Results saved to {filename}")

    return all_results_array
