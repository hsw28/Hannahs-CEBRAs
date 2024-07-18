import sys
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')

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
import argparse
import cebra.helper as cebra_helper
from consistency import consistency
import matplotlib.pyplot as plt
import joblib as jl
from matplotlib.collections import LineCollection

#ex: python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_script.py ./traceAnB1_An.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat 2 0 --iterations 2 --parameter_set_name test

#ex: python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_consistencyAB_script.py ./traceAn.mat ./traceB1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat 2 0 --iterations 2 --parameter_set_name test

# This function measures consistency across environments for the same rat
def cond_consistencyAB(envA_cell_train, envB_cell_train, envA_eyeblink, envB_eyeblink, iterations, parameter_set):
    learning_rate = parameter_set["learning_rate"]
    min_temperature = parameter_set["min_temperature"]
    max_iterations = parameter_set["max_iterations"]
    distance = parameter_set["distance"]
    temp_mode = parameter_set["temp_mode"]

    results = np.zeros((iterations, 4))

    # Consistency between two models
    if temp_mode == 'auto':
        cebra_loc_model = CEBRA(model_architecture='offset10-model',
                                batch_size=512,
                                learning_rate=learning_rate,
                                temperature_mode='auto',
                                min_temperature=min_temperature,
                                output_dimension=3,
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
                                output_dimension=3,
                                max_iterations=max_iterations,
                                distance=distance,
                                conditional='time_delta',
                                device='cuda_if_available',
                                num_hidden_units=32,
                                time_offsets=1,
                                verbose=False)

    fract_control_all = []
    fract_test_all = []

    envs_cell_train = [envA_cell_train, envB_cell_train]
    envs_eyeblink = [envA_eyeblink, envB_eyeblink]

    min_length = min(len(data) for data in envs_eyeblink)
    if min_length % 10 == 9:
        envs_eyeblink = [data[9:] for data in envs_eyeblink]
        envs_cell_train = [data[9:] for data in envs_cell_train]

    envA_cell_train = envs_cell_train[0]
    envB_cell_train = envs_cell_train[1]
    envA_eyeblink = envs_eyeblink[0]
    envB_eyeblink = envs_eyeblink[1]

    for i in range(iterations):
        print(i)
        # First unshuffled
        eyeblink_train_controlA = envA_eyeblink
        cell_train_controlA = envA_cell_train
        #eyeblink_train_controlA, eyeblink_train_controlA = hold_out(envA_eyeblink, 1)
        #cell_train_controlA, cell_train_controlA = hold_out(envA_cell_train, 1)

        eyeblink_train_controlB = envB_eyeblink
        cell_train_controlB = envB_cell_train
        #eyeblink_train_controlB, eyeblink_train_controlB = hold_out(envB_eyeblink, 1)
        #cell_train_controlB, cell_train_controlB = hold_out(envB_cell_train, 1)

        if not np.array_equal(eyeblink_train_controlA[:10], eyeblink_train_controlB[:10]):
            min_length = min(len(eyeblink_train_controlA), len(eyeblink_train_controlB))
            eyeblink_train_controlA, eyeblink_train_controlB = eyeblink_train_controlA[:min_length], eyeblink_train_controlB[:min_length]
            cell_train_controlA, cell_train_controlB = cell_train_controlA[:min_length], cell_train_controlB[:min_length]

        model1 = cebra_loc_model.fit(envA_cell_train, envA_eyeblink).transform(cell_train_controlA)
        model2 = cebra_loc_model.fit(envB_cell_train, envB_eyeblink).transform(cell_train_controlB)


        scores_runs, pairs_runs, ids_runs = consistency([model1, model2])

        # Now shuffled

        shuffled_index_A = np.random.permutation(envA_cell_train.shape[0])
        envA_cell_train_shuff = envA_cell_train[shuffled_index_A, :]

        shuffled_index_B = np.random.permutation(envB_cell_train.shape[0])
        envB_cell_train_shuff = envB_cell_train[shuffled_index_B, :]

        eyeblink_train_controlA = envA_eyeblink
        cell_train_controlA = envA_cell_train_shuff
        #eyeblink_train_controlA, eyeblink_train_controlA = hold_out(envA_eyeblink, 1)
        #cell_train_controlA, cell_train_controlA = hold_out(envA_cell_train_shuff, 1)

        eyeblink_train_controlB = envB_eyeblink
        cell_train_controlB = envB_cell_train_shuff
        #eyeblink_train_controlB, eyeblink_train_controlB = hold_out(envB_eyeblink, 1)
        #cell_train_controlB, cell_train_controlB = hold_out(envB_cell_train_shuff, 1)

        if not np.array_equal(eyeblink_train_controlA[:10], eyeblink_train_controlB[:10]):
            min_length = min(len(eyeblink_train_controlA), len(eyeblink_train_controlB))
            eyeblink_train_controlA, eyeblink_train_controlB = eyeblink_train_controlA[:min_length], eyeblink_train_controlB[:min_length]
            cell_train_controlA, cell_train_controlB = cell_train_controlA[:min_length], cell_train_controlB[:min_length]


        model1_shuff = cebra_loc_model.fit(envA_cell_train, envA_eyeblink).transform(cell_train_controlA)
        model2_shuff = cebra_loc_model.fit(envB_cell_train, envB_eyeblink).transform(cell_train_controlB)

        scores_runs_shuff, pairs_runs_shuff, ids_runs_shuff = consistency([model1_shuff, model2_shuff])

        #results[i] = np.concatenate((np.ravel(scores_runs), np.ravel(pairs_runs), np.ravel(ids_runs), np.ravel(scores_runs_shuff), np.ravel(pairs_runs_shuff), np.ravel(ids_runs_shuff)))
        results[i] = np.concatenate((np.ravel(scores_runs), np.ravel(scores_runs_shuff)))

        try:
            print(results[i])
        except Exception as e:
            print(f"An error occurred: {e}")

    # Save the results to a CSV file with the current date and time in the filename
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"condAvsBconsistancy_lr{learning_rate}_mt{min_temperature}_mi{max_iterations}_d{distance}_mode{temp_mode}_{current_time}.csv"
    np.savetxt(filename, results, delimiter=',', fmt='%.3f')  # Adjust precision as needed
    print(f"Results saved to {filename}")

    return results
