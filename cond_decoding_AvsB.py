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

#decodes conditioning in envB using envA.
#Outputs percent correct in envA after being trained in env A(based on a 70/30 split)
#Outputs percent correct in envB using the model trained in envA



output_dimension = 2 #here, we set as a variable for hypothesis testing below.
cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        #learning_rate= 5e-6,
                        learning_rate= 8.6e-4,
                        temperature_mode = 'auto',
                        min_temperature = .2, #<---------------.3
                        #temperature = .5,
                        output_dimension=output_dimension,
                        max_iterations=15000, #<--------------1-20000
                        #distance='euclidean',
                        distance='cosine',
                        conditional='time_delta', #added, keep
                        device='cuda_if_available',
                        num_hidden_units = 32,
                        time_offsets = 1,
                        verbose=True)


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


results21 = np.zeros((5, 1))


# Loop to run the batch of code 50 times
for i in range(5):
  print(i)

  ######### use this to test in own environment
  eyeblink_train = trainingtime22[:350]
  cell_train = trainingcells22[:350,:]
  cell_test = trainingcells22[150:,:]

  #########
  #define variables
  eyeblink_train = trainingtime22
  eyeblink_test = trainingtime21
  cell_train = trainingcells22_21
  cell_test = trainingcells21

  #run the model
  cebra_loc_model.fit(cell_train, eyeblink_train)

  #determine model fit
  cebra_loc_test22 = cebra_loc_model.transform(cell_test)
  cebra_loc_train22 = cebra_loc_model.transform(cell_train)


  fract21, CSUS_test_err, CSUS_test_score, abs_dif, predicted, actual, test_score, pos_test_err, pos_test_score, dis_mean, dis_median = decoding_CSUS(cebra_loc_train22, cebra_loc_test22, eyeblink_train, eyeblink_test)
  print(fract21)
