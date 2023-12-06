#POSITION DECODING USING OWN CELLS

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import numpy as np
import torch
import random
import cebra
from cebra import CEBRA
import matplotlib.pyplot as plt
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection



SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


########################

eyeblink_train = pos21
cell_train = trace22_all[0:5500,:]
cell_test = trace22_all[5500:,:]


output_dimension = 3 #here, we set as a variable for hypothesis testing below.
cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        #learning_rate= 3e-4,
                        learning_rate= 5e-6,
                        #temperature = 2,
                        temperature_mode = 'auto',
                        #min_temperature = .74,
                        output_dimension=output_dimension,
                        max_iterations=9000,
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
                        max_iterations=9000,
                        distance='euclidean',
                        conditional='time_delta', #added, keep
                        device='cuda_if_available',
                        num_hidden_units = 10,
                        time_offsets = 1,
                        #hybrid=True, #added <-- if using time
                        verbose=True)

def decoding_pos_dir(emb_train, emb_test, label_train, label_test, n_neighbors=32):
    pos_decoder = KNeighborsRegressor(n_neighbors, metric = 'cosine')
    pos_decoder.fit(emb_train, label_train)
    pos_pred = pos_decoder.predict(emb_test)
    prediction = pos_pred
    test_score = r2_score(label_test, prediction)
    pos_test_err = np.median(abs(prediction - label_test))
    pos_test_score = r2_score(label_test, prediction)
    # Compute the squared differences for each dimension
    squared_diffs = (prediction - label_test) ** 2
    # Sum the squared differences across columns (axis=1) and take the square root
    distances = np.sqrt(np.sum(squared_diffs, axis=1))
    dis_mean = (np.mean(distances))
    dis_median = (np.median(distances))


    return test_score, pos_test_err, pos_test_score, dis_mean, dis_median

########################

eyeblink_train = pos21
cell_train = trace21

cebra_loc_model.fit(cell_train, eyeblink_train)
cebra_loc_train21 = cebra_loc_model.transform(cell_train)

###########
eyeblink_train = pos22
cell_train = traceA22B24_22

cebra_loc_model.fit(cell_train, eyeblink_train)
cebra_loc_train22 = cebra_loc_model.transform(cell_train)

#############

eyeblink_train = pos24
cell_train = trace24


cebra_loc_model.fit(cell_train, eyeblink_train)
cebra_loc_train24 = cebra_loc_model.transform(cell_train)
##############


eyeblink_train = pos25
cell_train = trace25

cebra_loc_model.fit(cell_train, eyeblink_train)
cebra_loc_train25 = cebra_loc_model.transform(cell_train)


print(cebra_loc_train21.shape)
print(cebra_loc_train22.shape)
print(cebra_loc_train24.shape)
print(cebra_loc_train25.shape)
