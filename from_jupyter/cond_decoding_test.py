#TESTING METRICS FOR CONDITIONING

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score
import sklearn
import numpy as np
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


######

eyeblink_train = trainingtime22[:350]
cell_train = trainingcells22[:350,:]
cell_test = trainingcells22[150:,:]

########################

output_dimension = 2 #here, we set as a variable for hypothesis testing below.
cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate= 5e-6,
                        temperature_mode = 'auto',
                        min_temperature = .74,
                        output_dimension=output_dimension,
                        max_iterations=5000,
                        distance='euclidean',
                        conditional='time_delta', #added, keep
                        device='cuda_if_available',
                        num_hidden_units = 10,
                        time_offsets = 1,
                        #hybrid=True, #added <-- if using time
                        verbose=True)


cebra_loc_model.fit(cell_train, eyeblink_train)

cebra_loc_test22 = cebra_loc_model.transform(cell_test)
cebra_loc_train22 = cebra_loc_model.transform(cell_train)



def decoding_CSUS(emb_train, emb_test, label_train, label_test, n_neighbors=32):
    CSUS_decoder = KNeighborsClassifier(n_neighbors, metric = 'cosine')
    CSUS_decoder.fit(emb_train, label_train)


    pos_pred = CSUS_decoder.predict(emb_test)
    prediction = pos_pred
    test_score = r2_score(label_test, prediction)
    pos_test_err = np.median(abs(prediction - label_test))
    pos_test_score = r2_score(label_test, prediction)
    # Compute the squared differences for each dimension
    squared_diffs = (prediction - label_test) ** 2
    # Sum the squared differences across columns (axis=1) and take the square root
    distances = np.sqrt(np.sum(squared_diffs, axis=0))
    dis_mean = (np.mean(distances))
    dis_median = (np.median(distances))

    predicted = CSUS_decoder.predict(emb_test)
    actual = label_test
    CSUS_test_err = np.median(abs(predicted[:] - label_test[:]))
    CSUS_test_score = sklearn.metrics.r2_score(label_test[:], predicted[:])
    dif = (predicted.astype('int32') - label_test.astype('int32'))
    abs_dif = np.abs(dif)
    num_zeros = np.sum(abs_dif == 0)  # Count the number of zeros in abs_dif
    total_values = len(abs_dif)  # Get the total number of values in abs_dif
    fract = num_zeros / total_values
    abs_dif = np.mean(abs_dif)

    return fract, CSUS_test_err, CSUS_test_score, abs_dif, predicted, actual, test_score, pos_test_err, pos_test_score, dis_mean, dis_median


#test_score, pos_test_err, pos_test_score, dis_mean, dis_median = decoding_pos_dir(cebra_loc_train21, cebra_loc_test21, pos22[:,1:], pos21[:,1:])
#print(dis_mean)
#test_score, pos_test_err, pos_test_score, dis_mean, dis_median = decoding_pos_dir(cebra_loc_train24, cebra_loc_test24, pos22[:,1:], pos24[:,1:])
#print(dis_mean)
#test_score, pos_test_err, pos_test_score, dis_mean, dis_median = decoding_pos_dir(cebra_loc_train25, cebra_loc_test25, pos22[:,1:], pos25[:,1:])
#print(dis_mean)
fract, CSUS_test_err, CSUS_test_score, abs_dif, predicted, actual, test_score, pos_test_err, pos_test_score, dis_mean, dis_median = decoding_CSUS(cebra_loc_train22, cebra_loc_test22, trainingtime22[:350], trainingtime22[150:])
print(fract)
