import sys
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')
import argparse
import numpy as np
from itertools import product
import gc  # Garbage collection
import cebra
from cebra import CEBRA
import cebra.helper as cebra_helper
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score
import sklearn
import numpy as np
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import cebra
from cebra import CEBRA
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection
from CSUS_score import CSUS_score
from hold_out import hold_out
import gc



def predict_correct_grid(envA_cell_train, envB_cell_train, envA_eyeblink, envB_eyeblink, learning_rates, min_temperatures, max_iterations_list):
    results = []
    dimensions = 3
    for lr, temp, max_iter in product(learning_rates, min_temperatures, max_iterations_list):
        #print({'learning_rate': lr, 'min_temperature': temp, 'max_iterations': max_iter})
        # Setup the CEBRA model with the current set of parameters
        cebra_loc_model = CEBRA(
            learning_rate=lr,
            max_iterations=max_iter,
            model_architecture='offset10-model',
            batch_size=512,
            temperature_mode='auto',
            min_temperature=temp,
            output_dimension=3,
            distance='cosine',
            conditional='time_delta',
            device='cuda_if_available',
            num_hidden_units=32,
            time_offsets=1,
            verbose=False
        )

        fract_control_all = []
        fract_test_all = []
        loss_all = []

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

        # Loop to run the batch of code 50 times
        for i in range(1):



              #test control environment

              ######### use this to test in own environment
              eyeblink_train_control, eyeblink_test_control = hold_out(envA_eyeblink, .75)
              cell_train_control, cell_test_control  = hold_out(envA_cell_train,.75)

              #run the model
              cebra_loc_modelpos = cebra_loc_model.fit(cell_train_control, eyeblink_train_control)
              loss_all.append(cebra_loc_model.state_dict_['loss'][-1])
              #determine model fit
              cebra_loc_train22 = cebra_loc_modelpos.transform(cell_train_control)
              cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test_control)


              #find fraction correct
              fract_controlA = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)


              #test with using A to decode B
              cell_test = envB_cell_train
              eyeblink_test_control = envB_eyeblink

              #if i want to fit B using fulling training, but i think i want to do it with held out
              '''
              cebra_loc_modelpos_full = cebra_loc_model.fit(envA_cell_train, envA_eyeblink)
              loss_all.append(cebra_loc_model.state_dict_['loss'][-1])
              #determine model fit
              cebra_loc_train22 = cebra_loc_modelpos.transform(envA_cell_train)
              cebra_loc_test22 = cebra_loc_modelpos_full.transform(cell_test)
              #find fraction correct
              fract_testB = CSUS_score(cebra_loc_train22, cebra_loc_test22, envA_eyeblink, eyeblink_test_control)
              '''


              #determine model fit
              cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test)
              #find fraction correct
              fract_testB = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)

              fract_controlA = round(fract_controlA,3)
              fract_testB = round(fract_testB,3)

              fract_control_all.append(fract_controlA)
              fract_test_all.append(fract_testB)


              del cebra_loc_modelpos, cebra_loc_train22, cebra_loc_test22
              gc.collect()

              #print((fract_control_all))
              #print((fract_test_all))

        # Calculate mean of all fractions
        mean_control = np.mean(fract_control_all)
        mean_test = np.mean(fract_test_all)  # Corrected to use fract_test_all
        mean_loss = np.mean(loss_all)
        std_loss = np.std(loss_all)


        # Round the mean values
        mean_control = round(mean_control, 3)
        mean_test = round(mean_test, 3)
        mean_loss = round(mean_loss,3)
        std_loss = round(std_loss,3)

        # Append the correctly calculated means to the results
        results.append({
            'learn_rate': lr,
            'min_temp': temp,
            'max_it': max_iter,
        #    'fract_control': fract_control_all,
        #    'fract_test': fract_test_all,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'mean_control': mean_control,  # Correctly calculated mean
            'mean_test': mean_test         # Correctly calculated mean
        })

        print(results)
    return results
