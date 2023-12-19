###### COMPARING CONDITIONING
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


def cond_decoding_AvsB(envA_cell_train, envA_eyeblink, envB_cell_train, envB_eyeblink):
    output_dimension = 2 #here, we set as a variable for hypothesis testing below.
    cebra_loc_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            #learning_rate= 8.6e-2,
                            learning_rate= 8.6e-4,
                            temperature_mode = 'auto',
                            min_temperature = .2,
                            #temperature = .2,
                            output_dimension=output_dimension,
                            max_iterations=15000, #<--------------1-20000
                            #max_iterations=15000, #<--------------1-20000
                            distance='cosine',
                            conditional='time_delta', #added, keep
                            device='cuda_if_available',
                            num_hidden_units = 32,
                            time_offsets = 1,
                            verbose=True)
'''
%first
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

65
cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        #learning_rate= 5e-6,
                        learning_rate= 8.6e-4,
                        temperature_mode = 'auto',
                        min_temperature = .8, #.5 best so far i think = 61 .8=60.2,
                        #temperature = .5,
                        output_dimension=output_dimension,
                        max_iterations=5500, #<--------------8000=60.2, 7000=60.6, 6000=61.6 5000=60.8
                        distance='euclidean',
                        #distance='cosine',
                        conditional='time_delta', #added, keep
                        device='cuda_if_available',
                        num_hidden_units = 32,
                        time_offsets = 1,
                        verbose=True)
#63.8

cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        #learning_rate= 5e-6,
                        learning_rate= 8e-4,
                        temperature_mode = 'auto',
                        min_temperature = .5, #.5 best so far i think = 61
                        #temperature = .5,
                        output_dimension=output_dimension,
                        max_iterations=8000, #<--------------64 with 7000
                        distance='euclidean',
                        #distance='cosine',
                        conditional='time_delta', #added, keep
                        device='cuda_if_available',
                        num_hidden_units = 30,
                        time_offsets = 1,
                        #hybrid=True, #added <-- if using time
                        verbose=True)

#63.5
cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        #learning_rate= 5e-6,
                        learning_rate= 8.7e-4,
                        #temperature_mode = 'auto',
                        #min_temperature = .5, #.5 best so far i think = 61
                        temperature = .7,
                        output_dimension=output_dimension,
                        max_iterations=7200,
                        distance='euclidean',
                        #distance='cosine',
                        conditional='time_delta', #added, keep
                        device='cuda_if_available',
                        num_hidden_units = 30,
                        time_offsets = 1,
                        #hybrid=True, #added <-- if using time
                        verbose=True)

'''

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
results24 = np.zeros((5, 1))
results21_shuff = np.zeros((5, 1))
results24_shuff = np.zeros((5, 1)) #results24_shuff = np.zeros((10, 1))

# Mount your Google Drive (do this outside the loop, only once)
drive.mount('/content/drive')

# Define the path to the folder in your Google Drive where you want to save the files
path = '/content/drive/MyDrive/Colab Notebooks'

# Loop to run the batch of code 50 times
for i in range(5):
  print(i)

  # Create a figure and a 2x3 grid of subplots
  #fig, axs = plt.subplots(1, 4, figsize=(40, 10))  # Adjust figsize as needed
  fig, axs = plt.subplots(2, 4, figsize=(28, 10))

  #########
  eyeblink_train = trainingtime22
  eyeblink_test = trainingtime21
  cell_train = trainingcells22_21
  cell_test = trainingcells21


  cebra_loc_model.fit(cell_train, eyeblink_train)

  #fig = plt.figure(figsize=(5,5))
  #ax = plt.subplot(111)
  #ax.plot(cebra_loc_model.state_dict_['loss'], c='deepskyblue', label = 'position+direction')

  cebra_loc_test22 = cebra_loc_model.transform(cell_test)
  cebra_loc_train22 = cebra_loc_model.transform(cell_train)


  plot_hippocampus2d(axs[0,0], cebra_loc_train22, eyeblink_train, eyeblink_train)
  plot_hippocampus2d(axs[0,1], cebra_loc_test22, eyeblink_test, eyeblink_test)

  fract21, CSUS_test_err, CSUS_test_score, abs_dif, predicted, actual, test_score, pos_test_err, pos_test_score, dis_mean, dis_median = decoding_CSUS(cebra_loc_train22, cebra_loc_test22, eyeblink_train, eyeblink_test)
  print(fract21)





  ############
  eyeblink_train = trainingtime22
  eyeblink_test = trainingtime24
  cell_train = trainingcells22_24
  cell_test = trainingcells24


  cebra_loc_model.fit(cell_train, eyeblink_train)
  cebra_loc_test22 = cebra_loc_model.transform(cell_test)
  cebra_loc_train22 = cebra_loc_model.transform(cell_train)

  fract24, CSUS_test_err, CSUS_test_score, abs_dif, predicted, actual, test_score, pos_test_err, pos_test_score, dis_mean, dis_median = decoding_CSUS(cebra_loc_train22, cebra_loc_test22, eyeblink_train, eyeblink_test)
  print(fract24)


  plot_hippocampus2d(axs[0,2], cebra_loc_train22, eyeblink_train, eyeblink_train)
  plot_hippocampus2d(axs[0,3], cebra_loc_test22, eyeblink_test, eyeblink_test)


  ############
  '''
  eyeblink_train = trainingtime22
  eyeblink_test = trainingtime25
  cell_train = trainingcells22_25
  cell_test = trainingcells25


  cebra_loc_model.fit(cell_train, eyeblink_train)
  cebra_loc_test22 = cebra_loc_model.transform(cell_test)
  cebra_loc_train22 = cebra_loc_model.transform(cell_train)

  fract, CSUS_test_err, CSUS_test_score, abs_dif, predicted, actual, test_score, pos_test_err, pos_test_score, dis_mean, dis_median = decoding_CSUS(cebra_loc_train22, cebra_loc_test22, eyeblink_train, eyeblink_test)
  print(fract)

  plot_hippocampus2d(axs[0,3], cebra_loc_test22, eyeblink_test, eyeblink_test)
  '''


  ################ SHUFFLING 22/21

  eyeblink_train = trainingtime22
  eyeblink_test = trainingtime21
  cell_train = trainingcells22_21
  cell_test = trainingcells21

  eyeblink_train = trainingtime22
  # Convert to numpy array if not already
  eyeblink_train = np.array(eyeblink_train)
  # Create a new array to hold the shuffled data
  eyeblink_shuff = eyeblink_train.copy()
  # Shuffle each column independently
  np.random.shuffle(eyeblink_shuff[:])

  cebra_loc_model.fit(cell_train, eyeblink_shuff)

  #fig = plt.figure(figsize=(5,5))
  #ax = plt.subplot(111)
  #ax.plot(cebra_loc_model.state_dict_['loss'], c='deepskyblue', label = 'position+direction')

  cebra_loc_test22 = cebra_loc_model.transform(cell_test)
  cebra_loc_train22 = cebra_loc_model.transform(cell_train)


  plot_hippocampus2d(axs[1,0], cebra_loc_train22, eyeblink_train, eyeblink_train)
  plot_hippocampus2d(axs[1,1], cebra_loc_test22, eyeblink_test, eyeblink_test)

  fract21_shuff, CSUS_test_err, CSUS_test_score, abs_dif, predicted, actual, test_score, pos_test_err, pos_test_score, dis_mean, dis_median = decoding_CSUS(cebra_loc_train22, cebra_loc_test22, eyeblink_shuff, eyeblink_test)
  print(fract21_shuff)


  ################ SHUFFLING 22/24

  eyeblink_test = trainingtime24
  cell_train = trainingcells22_24
  cell_test = trainingcells24

  cebra_loc_model.fit(cell_train, eyeblink_shuff)
  cebra_loc_test22 = cebra_loc_model.transform(cell_test)
  cebra_loc_train22 = cebra_loc_model.transform(cell_train)

  fract24_shuff, CSUS_test_err, CSUS_test_score, abs_dif, predicted, actual, test_score, pos_test_err, pos_test_score, dis_mean, dis_median = decoding_CSUS(cebra_loc_train22, cebra_loc_test22, eyeblink_shuff, eyeblink_test)
  print(fract24_shuff)

  plot_hippocampus2d(axs[1,2], cebra_loc_train22, eyeblink_train, eyeblink_train)
  plot_hippocampus2d(axs[1,3], cebra_loc_test22, eyeblink_test, eyeblink_test)


  ############## shufftling 22/25
  '''

  eyeblink_test = trainingtime25
  cell_train = trainingcells22_25
  cell_test = trainingcells25


  cebra_loc_model.fit(cell_train, eyeblink_shuff)
  cebra_loc_test22 = cebra_loc_model.transform(cell_test)
  cebra_loc_train22 = cebra_loc_model.transform(cell_train)

  fract, CSUS_test_err, CSUS_test_score, abs_dif, predicted, actual, test_score, pos_test_err, pos_test_score, dis_mean, dis_median = decoding_CSUS(cebra_loc_train22, cebra_loc_test22, eyeblink_shuff, eyeblink_test)
  print(fract)

  plot_hippocampus2d(axs[1,3], cebra_loc_test22, eyeblink_test, eyeblink_test)

  '''
  #plt.savefig('my_figure.svg', format='svg')
  results21[i] = fract21
  results24[i] = fract24
  results21_shuff[i] = fract21_shuff
  results24_shuff[i] = fract24_shuff

  # Save each figure with a unique name including the iteration number
  figure_name = f'my_figure_{i}.svg'
  plt.savefig(figure_name, format='svg')

  # Define the source and destination paths
  source_path = f"/content/{figure_name}"
  destination_path = f"{path}{figure_name}"

  # Copy the figure to your Google Drive using shutil
  shutil.copy(source_path, destination_path)



print(results21)
print(results24)
print(results21_shuff)
print(results24_shuff)


# Save the results to a CSV file using numpy
np.savetxt("results21.csv", results21, delimiter=",")
np.savetxt("results24.csv", results24, delimiter=",")
np.savetxt("results21_shuff.csv", results21_shuff, delimiter=",")
np.savetxt("results24_shuff.csv", results24_shuff, delimiter=",")



'''
# Copy the files to your Google Drive
!cp "/content/results21.csv" "{path}" ########
!cp "/content/results24.csv" "{path}"###########
!cp "/content/results21_shuff.csv" "{path}" ########
!cp "/content/results24_shuff.csv" "{path}"###########
