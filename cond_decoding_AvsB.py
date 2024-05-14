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




#decodes conditioning in envB using envA.
#Outputs percent correct in envA after being trained in env A(based on a 75/25 split)
#Outputs percent correct in envB using the model trained in envA


def cond_decoding_AvsB(envA_cell_train, envA_eyeblink, envB_cell_train, envB_eyeblink, dimensions=2):

    output_dimension = dimensions  #here, we set as a variable for hypothesis testing below.
    cebra_loc_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            #learning_rate= .046,
                            learning_rate= .001,
                            temperature_mode = 'auto',
                            #min_temperature = .2,
                            output_dimension=output_dimension,
                            max_iterations=15000, #<--------------1-20000
                            distance='cosine',
                            conditional='time_delta', #added, keep
                            device='cuda_if_available',
                            num_hidden_units = 32,
                            time_offsets = 1,
                            verbose='true')



    fract_control_all = []
    fract_test_all = []

    # Loop to run the batch of code 50 times
    for i in range(1):

          #test control environment


          ######### use this to test in own environment
          eyeblink_train_control, eyeblink_test_control = hold_out(envA_eyeblink, .75)
          cell_train_control, cell_test_control  = hold_out(envA_cell_train,.75)

          #run the model
          cebra_loc_modelpos = cebra_loc_model.fit(cell_train_control, eyeblink_train_control)
          #determine model fit
          cebra_loc_train22 = cebra_loc_modelpos.transform(cell_train_control)
          cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test_control)


          #find fraction correct
          fract_controlA = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)



          #test with using A to decode B
          cell_test = envB_cell_train
          eyeblink_test_control = envB_eyeblink

          cebra_loc_modelpos_full = cebra_loc_model.fit(envA_cell_train, envA_eyeblink)

          #determine model fit
          cebra_loc_train22 = cebra_loc_modelpos.transform(envA_cell_train)
          cebra_loc_test22 = cebra_loc_modelpos_full.transform(cell_test)
          #find fraction correct
          fract_testB = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)


          fract_control_all.append(fract_controlA)
          fract_test_all.append(fract_testB)


    del cebra_loc_modelpos, cebra_loc_train22, cebra_loc_test22
    gc.collect()

    print((fract_control_all))
    print((fract_test_all))

    return fract_control_all, fract_test_all
