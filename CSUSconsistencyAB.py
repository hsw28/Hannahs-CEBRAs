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
import argparse
import cebra.helper as cebra_helper
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')
from consistency import consistency
import matplotlib.pyplot as plt
import joblib as jl
from matplotlib.collections import LineCollection

###this measures consistancy across environments for the same rat


def CSUSconsistencyAB(envA_cell_train, envB_cell_train, envA_eyeblink, envB_eyeblink, iterations, parameter_set):

    learning_rate = parameter_set["learning_rate"]
    min_temperature = parameter_set["min_temperature"]
    max_iterations = parameter_set["max_iterations"]
    distance = parameter_set["distance"]
    temp_mode = parameter_set["temp_mode"]

    #looks at consistancy between two models
    if temperature_mode='auto':
        cebra_loc_model = CEBRA(model_architecture='offset10-model',
                                batch_size=512,
                                learning_rate= learning_rate,
                                temperature_mode = 'auto',
                                min_temperature = temperature,
                                output_dimension=3,
                                max_iterations= max_iterations
                                distance= distance,
                                conditional='time_delta',
                                device='cuda_if_available',
                                num_hidden_units = 32,
                                time_offsets = 1,
                                verbose='true')

    if temperature_mode='constant':
        cebra_loc_model = CEBRA(model_architecture='offset10-model',
                                batch_size=512,
                                learning_rate= learning_rate,
                                temperature_mode = 'constant',
                                temperature = temperature,
                                output_dimension=3,
                                max_iterations=max_iterations
                                distance= distance,
                                conditional='time_delta',
                                device='cuda_if_available',
                                num_hidden_units = 32,
                                time_offsets = 1,
                                verbose='true')


    fract_control_all = []
    fract_test_all = []


    for i in range(iterations):
        #first unshuffled
        eyeblink_train_controlA, eyeblink_test_controlA = hold_out(envA_eyeblink, .75)
        cell_train_controlA, cell_test_controlA  = hold_out(envA_cell_train,.75)

        eyeblink_train_controlB, eyeblink_test_controlB = hold_out(envB_eyeblink, .75)
        cell_train_controlB, cell_test_controlB  = hold_out(envB_cell_train,.75)

        if not np.array_equal(eyeblink_test_controlA[:10], eyeblink_test_controlB[:10]):
            # Determine the amount to truncate from the training sets to make the first 10 values the same
            min_length = min(len(eyeblink_test_controlA), len(eyeblink_test_controlB))
            eyeblink_test_controlA, eyeblink_test_controlB = eyeblink_test_controlA[:min_length], eyeblink_test_controlB[:min_length]
            # Truncate traceA and traceB to maintain alignment with the truncated training sets
            cell_test_controlA, cell_test_controlB = cell_test_controlA[:min_length], cell_test_controlB[:min_length]

        model1 = cebra_loc_model.fit(envA_cell_train, envA_eyeblink)
        model1 = model1.transform(cell_test_controlA)

        model2 = cebra_loc_model.fit(envB_cell_train, envB_eyeblink)
        model2 = model2.transform(cell_test_controlB)

        scores_runs, pairs_runs = consistency([model1, model2])
        print(model_consist) #model not shuffled

        #now shuffled... think I have to shuffle the cell trains so i can keep the eyeblink numbers in order
        #must shuffle them consistantly keeping columns together

        shuffled_index = np.random.permutation(envA_cell_train.index)
        envA_cell_train_shuff = envA_cell_train.reindex(shuffled_index)
        shuffled_index = np.random.permutation(envB_cell_train.index)
        envB_cell_train_shuff = envA_cell_train.reindex(shuffled_index)

        eyeblink_train_controlA, eyeblink_test_controlA = hold_out(envA_eyeblink, .75)
        cell_train_controlA, cell_test_controlA  = hold_out(envA_cell_train_shuff,.75)

        eyeblink_train_controlB, eyeblink_test_controlB = hold_out(envB_eyeblink, .75)
        cell_train_controlB, cell_test_controlB  = hold_out(envB_cell_train_shuff,.75)

        if not np.array_equal(eyeblink_test_controlA[:10], eyeblink_test_controlB[:10]):
            # Determine the amount to truncate from the training sets to make the first 10 values the same
            min_length = min(len(eyeblink_test_controlA), len(eyeblink_test_controlB))
            eyeblink_test_controlA, eyeblink_test_controlB = eyeblink_test_controlA[:min_length], eyeblink_test_controlB[:min_length]
            # Truncate traceA and traceB to maintain alignment with the truncated training sets
            cell_test_controlA, cell_test_controlB = cell_test_controlA[:min_length], cell_test_controlB[:min_length]

        model1_shuff = cebra_loc_model.fit(envA_cell_train, envA_eyeblink)
        model1_shuff = model1.transform(cell_test_controlA)

        model2_shuff = cebra_loc_model.fit(envB_cell_train, envB_eyeblink)
        model2_shuff = model2.transform(cell_test_controlB)

        scores_runs_shuff, pairs_runs_shuff = consistency([model1_shuff, model2_shuff])
        print(model_consist_shuff)

        #need to save scores_runs, pairs_runs and scores_runs_shuff, pairs_runs_shuff for each iteration

    return results
