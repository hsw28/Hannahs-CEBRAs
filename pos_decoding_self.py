from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score
from hold_out import hold_out
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
from pos_score import pos_score

#decodes own position using held out data and compares to shuffled

def pos_decoding_self(cell_trace, pos, percent_to_train):

    output_dimension = 3 #here, we set as a variable for hypothesis testing below.
    cebra_loc_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            #learning_rate= 3e-4,
                            learning_rate= 5e-6,
                            #temperature = 2,
                            temperature_mode = 'auto',
                            #min_temperature = .74,
                            output_dimension=output_dimension,
                            max_iterations=8000,
                            distance='euclidean',
                            conditional='time_delta', #added, keep
                            device='cuda_if_available',
                            num_hidden_units = 10,
                            time_offsets = 1,
                            #hybrid=True, #added <-- if using time
                            verbose=True)

    shuff_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            #learning_rate= 3e-4,
                            learning_rate= 5e-6,
                            #temperature = 2,
                            temperature_mode = 'auto',
                            #min_temperature = .74,
                            output_dimension=output_dimension,
                            max_iterations=8000,
                            distance='euclidean',
                            conditional='time_delta', #added, keep
                            device='cuda_if_available',
                            num_hidden_units = 10,
                            time_offsets = 1,
                            #hybrid=True, #added <-- if using time
                            verbose=True)

    ########################
    #TEST


    cell_train, cell_test = hold_out(cell_trace, percent_to_train)
    pos_train, pos_test = hold_out(pos, percent_to_train)


    err_all = []
    err_all_shuff = []
    for i in range(1):
        cebra_loc_model.fit(cell_train, pos_train)
        cebra_loc_model.save("cebra_loc_model.pt")
        cebra_loc_train = cebra_loc_model.transform(cell_train)
        cebra_loc_test = cebra_loc_model.transform(cell_test)

        pos_test_score, pos_test_err, dis_mean, dis_median = pos_score(cebra_loc_train, cebra_loc_test, pos_train, pos_test)
        #want pos_test_err


        ########################
        #SHUFFLED

        # Create a new array to hold the shuffled data
        pos_train_shuff = pos_train.copy()
        # Shuffle each column independently
        for column in range(pos_train_shuff.shape[1]):
            np.random.shuffle(pos_train_shuff[:, column])

        # Fit the model with the shuffled data
        shuff_model.fit(cell_train, pos_train_shuff)
        shuff_model.save("shuff_model.pt")
        cebra_loc_train_shuff = shuff_model.transform(cell_train)
        cebra_loc_test_shuff = shuff_model.transform(cell_test)

        pos_test_score_shuff, pos_test_err_shuff, dis_mean_shuff, dis_median_shuff = pos_score(cebra_loc_train_shuff, cebra_loc_test_shuff, pos_train, pos_test)

        err_all.append(dis_median)
        err_all_shuff.append(dis_median_shuff)

    #print(np.mean(err_all))
    #print(np.mean(err_all_shuff))

    return os_test_score, pos_test_err, dis_mean, dis_median, pos_test_score_shuff, pos_test_err_shuff, dis_mean_shuff, dis_median_shuff
