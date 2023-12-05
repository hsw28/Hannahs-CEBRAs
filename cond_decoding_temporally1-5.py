##### COMPARING CONDITIONING FROm 1-5 temporally
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score
from google.colab import drive

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


output_dimension = 5 #here, we set as a variable for hypothesis testing below.
cebra_loc_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        #learning_rate= 5e-6,
                        learning_rate= 8.6e-4,
                        temperature_mode = 'auto',
                        #min_temperature = .2, #<---------------.3
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


fig, axs = plt.subplots(1, 2, figsize=(15, 5))

results21 = np.zeros((5, 1))
results24 = np.zeros((5, 1))
results21_shuff = np.zeros((5, 1))
results24_shuff = np.zeros((5, 1)) #results24_shuff = np.zeros((10, 1))

# Mount your Google Drive (do this outside the loop, only once)
#drive.mount('/content/drive')

# Define the path to the folder in your Google Drive where you want to save the files
path = '/content/drive/MyDrive/Colab Notebooks'

# Loop to run the batch of code 50 times
for i in range(1):
  print(i)

  # Create a figure and a 2x3 grid of subplots
  #fig, axs = plt.subplots(1, 4, figsize=(40, 10))  # Adjust figsize as needed


  #########
  eyeblink_train = trainingtime22
  eyeblink_test = trainingtime21
  cell_train = trainingcells22_21
  cell_test = trainingcells21


  cebra_loc_model.fit(cell_train, eyeblink_train)

  #fig = plt.figure(figsize=(5,5))
  #ax = plt.subplot(111)
  #ax.plot(cebra_loc_model.state_dict_['loss'], c='deepskyblue', label = 'position+direction')

  cebra_loc_test21 = cebra_loc_model.transform(cell_test)
  cebra_loc_train21 = cebra_loc_model.transform(cell_train)




  fract21, CSUS_test_err21, CSUS_test_score, abs_dif, predicted, actual = decoding_CSUS5(cebra_loc_train21, cebra_loc_test21, eyeblink_train, eyeblink_test)
  print(fract21)
  p21 = confmatrix(axs[0], actual, predicted)






  ############
  eyeblink_train = trainingtime22
  eyeblink_test = trainingtime24
  cell_train = trainingcells22_24
  cell_test = trainingcells24


  cebra_loc_model.fit(cell_train, eyeblink_train)
  cebra_loc_test24 = cebra_loc_model.transform(cell_test)
  cebra_loc_train24 = cebra_loc_model.transform(cell_train)

  fract24, CSUS_test_err24, CSUS_test_score, abs_dif, predicted24, actual24 = decoding_CSUS5(cebra_loc_train24, cebra_loc_test24, eyeblink_train, eyeblink_test)
  print(fract24)
  p24 = confmatrix(axs[1], actual24, predicted24)





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

  cebra_loc_test21_shuff = cebra_loc_model.transform(cell_test)
  cebra_loc_train21_shuff = cebra_loc_model.transform(cell_train)



  fract21_shuff, CSUS_test_err21_shuff, CSUS_test_score, abs_dif, predicted, actual = decoding_CSUS5(cebra_loc_train21_shuff, cebra_loc_test21_shuff, eyeblink_shuff, eyeblink_test)
  print(fract21_shuff)


  ################ SHUFFLING 22/24

  eyeblink_test = trainingtime24
  cell_train = trainingcells22_24
  cell_test = trainingcells24

  cebra_loc_model.fit(cell_train, eyeblink_shuff)
  cebra_loc_test24_shuff = cebra_loc_model.transform(cell_test)
  cebra_loc_train24_shuff = cebra_loc_model.transform(cell_train)

  fract24_shuff, CSUS_test_err24_shuff, CSUS_test_score, abs_dif, predicted, actual = decoding_CSUS5(cebra_loc_train24_shuff, cebra_loc_test24_shuff, eyeblink_shuff, eyeblink_test)
  print(fract24_shuff)

  '''
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

  plt.savefig('my_figure.svg', format='svg')
  results21[i] = fract21
  results24[i] = fract24
  #results21_shuff[i] = fract21_shuff
  #results24_shuff[i] = fract24_shuff

  # Save each figure with a unique name including the iteration number
  figure_name = f'my_figure_{i}.svg'
  plt.savefig(figure_name, format='svg')

  # Define the source and destination paths
  #source_path = f"/content/{figure_name}"
  #destination_path = f"{path}{figure_name}"

  # Copy the figure to your Google Drive using shutil
  #shutil.copy(source_path, destination_path)



#print(results21)
#print(results24)
#print(results21_shuff)
#print(results24_shuff)


# Save the results to a CSV file using numpy
np.savetxt("results21.csv", results21, delimiter=",")
np.savetxt("results24.csv", results24, delimiter=",")
np.savetxt("results21_shuff.csv", results21_shuff, delimiter=",")
np.savetxt("results24_shuff.csv", results24_shuff, delimiter=",")
