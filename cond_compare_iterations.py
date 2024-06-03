import sys
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs')

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
    prefixes = ["A1An_held_out", "A1", "B1An_held_out", "B1", "SHUFF_A1An_held_out", "SHUFF_A1", "SHUFF_B1An_held_out", "SHUFF_B1"]
    metrics = ["% correct"]
    headers = []

    for prefix in prefixes:
        for metric in metrics:
            headers.append(f"{prefix}_{metric}")

    return ','.join(headers)

def cond_compare_iterations(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, CSUSAn, CSUSA1, CSUSB1, dimensions, iterations, parameter_set):

    learning_rate = parameter_set["learning_rate"]
    min_temperature = parameter_set["min_temperature"]
    max_iterations = parameter_set["max_iterations"]
    distance = parameter_set["distance"]
    temp_mode = parameter_set["temp_mode"]

    dimensions = dimensions

    if temp_mode == 'auto':
        cebra_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            learning_rate=learning_rate,
                            min_temperature=min_temperature,
                            output_dimension=output_dimension,
                            max_iterations=max_iterations,
                            distance=distance,
                            conditional='time_delta',
                            device='cuda_if_available',
                            num_hidden_units=32,
                            time_offsets=1,
                            verbose=False)

    if temp_mode == 'constant':
            cebra_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            learning_rate=learning_rate,
                            temperature_mode=temp_mode,
                            temperature=min_temperature,
                            output_dimension=output_dimension,
                            max_iterations=max_iterations,
                            distance=distance,
                            conditional='time_delta',
                            device='cuda_if_available',
                            num_hidden_units=32,
                            time_offsets=1,
                            verbose=False)


    results = np.zeros((iterations, 8))  # Each iteration results in 8 outputs

    try:
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

            # Flatten results into a single row per iteration
            results[i] = np.concatenate((np.ravel(regular_A1), np.ravel(regular_B1), np.ravel(shuffled_A1), np.ravel(shuffled_B1)))

            print(results)

        # Save the results to a CSV file with the current date and time in the filename
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"cond_lr{learning_rate}_mt{min_temperature}_mi{max_iterations}_d{distance}_mode{temp_mode}_{current_time}.csv"
        header = generate_headers()
        np.savetxt(filename, results, delimiter=',', header=header, comments='', fmt='%.3f')  # Adjust precision as needed
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return results
