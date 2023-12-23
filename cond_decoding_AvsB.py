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


#decodes conditioning in envB using envA.
#Outputs percent correct in envA after being trained in env A(based on a 70/30 split)
#Outputs percent correct in envB using the model trained in envA


def cond_decoding_AvsB(envA_cell_train, envA_eyeblink, envB_cell_train, envB_eyeblink):
    output_dimension = 2 #here, we set as a variable for hypothesis testing below.
    cebra_loc_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            #learning_rate= 5e-6,
                            learning_rate= 8.6e-4,
                            temperature_mode = 'auto',
                            min_temperature = .2, #<---------------.3
                            #temperature = .5,
                            output_dimension=output_dimension,
                            max_iterations=13000, #<--------------1-20000
                            #max_iterations=150, #<--------------1-20000
                            #distance='euclidean',
                            distance='cosine',
                            conditional='time_delta', #added, keep
                            device='cuda_if_available',
                            num_hidden_units = 32,
                            time_offsets = 1,
                            verbose=True)





    fract_control_all = []
    fract_test_all = []

    # Loop to run the batch of code 50 times
    for i in range(1):

          #test control environment


          ######### use this to test in own environment
          eyeblink_train_control = envA_eyeblink[:350]
          eyeblink_test_control = envA_eyeblink[350:]
          cell_train_control = envA_cell_train[:350,:]
          cell_test_control = envA_cell_train[350:,:]

          #run the model
          cebra_loc_model.fit(cell_train_control, eyeblink_train_control)
          #determine model fit
          cebra_loc_test22 = cebra_loc_model.transform(cell_test_control)
          cebra_loc_train22 = cebra_loc_model.transform(cell_train_control)




          #find fraction correct
          fract_control = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)



          #test with using A to decode B
          eyeblink_train = envA_eyeblink
          eyeblink_test = envB_eyeblink
          cell_train = envA_cell_train
          cell_test = envB_cell_train

          #run the model
          cebra_loc_model.fit(cell_train, eyeblink_train)
          #determine model fit
          cebra_loc_test22 = cebra_loc_model.transform(cell_test)
          cebra_loc_train22 = cebra_loc_model.transform(cell_train)
          #find fraction correct
          fract_test = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train, eyeblink_test)


          fract_control_all.append(fract_control)
          fract_test_all.append(fract_test)

    print(np.mean(fract_control_all))
    print(np.mean(fract_test))

    return fract_control_all, fract_test_all
