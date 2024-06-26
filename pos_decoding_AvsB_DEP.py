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
import gc

###THIS MIGHT BE DEPRECATED WITH POS_COMPARE
#decodes own position using trace and pos from A, then uses it to decide pos from B, compares both to shuffled

def pos_decoding_AvsB_dep(cell_traceA, posA, cell_traceB, posB, percent_to_train):

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
                            #max_iterations=8000,
                            distance='euclidean',
                            conditional='time_delta', #added, keep
                            device='cuda_if_available',
                            num_hidden_units = 32,
                            time_offsets = 1,
                            #hybrid=True, #added <-- if using time
                            verbose=False)

    shuff_model = CEBRA(model_architecture='offset10-model',
                            batch_size=512,
                            #learning_rate= 3e-4,
                            learning_rate= 5e-6,
                            #temperature = 2,
                            temperature_mode = 'auto',
                            #min_temperature = .74,
                            output_dimension=output_dimension,
                            max_iterations=8000,
                            #max_iterations=8000,
                            distance='euclidean',
                            conditional='time_delta', #added, keep
                            device='cuda_if_available',
                            num_hidden_units = 32,
                            time_offsets = 1,
                            #hybrid=True, #added <-- if using time
                            verbose=False)

    ########################
    #TEST



    err_allA = [] * 4
    err_allB_usingA = [] * 4
    err_all_shuffB_usingA = [] * 4
    err_allB_usingB = [] * 4

    for i in range(1):
        cell_trainA, cell_testA = hold_out(cell_traceA, percent_to_train)
        pos_trainA, pos_testA = hold_out(posA, percent_to_train)

        cebra_loc_modelA = cebra_loc_model.fit(cell_trainA, pos_trainA) #training on A
        cebra_loc_trainA = cebra_loc_modelA.transform(cell_trainA) #training on A
        cebra_loc_testA = cebra_loc_modelA.transform(cell_testA) #testing on A


        pos_test_scoreA, pos_test_errA, dis_meanA, dis_medianA = pos_score(cebra_loc_trainA, cebra_loc_testA, pos_trainA, pos_testA)
        #want pos_test_err

        cebra_loc_testB = cebra_loc_modelA.transform(cell_traceB) #training on A, testing on A
        pos_test_scoreBwA, pos_test_errBwA, dis_meanBwA, dis_medianBwA = pos_score(cebra_loc_trainA, cebra_loc_testB, pos_trainA, posB)


        ########################
        #SHUFFLED

        # Create a new array to hold the shuffled data
        pos_train_shuffA = pos_trainA.copy()
        # Shuffle each column independently
        for column in range(pos_train_shuffA.shape[1]):
            np.random.shuffle(pos_train_shuffA[:, column])

        # Fit the model with the shuffled data
        shuff_modelA = cebra_loc_model.fit(cell_trainA, pos_train_shuffA) #training on shuffled A
        cebra_loc_train_shuffA = shuff_modelA.transform(cell_trainA) #training on A
        cebra_loc_test_shuffA = shuff_modelA.transform(cell_testA) #testing on A

        pos_test_score_shuffA, pos_test_err_shuffA, dis_mean_shuffA, dis_median_shuffA = pos_score(cebra_loc_train_shuffA, cebra_loc_test_shuffA, pos_trainA, pos_testA)



        cebra_loc_test_shuffB = shuff_modelA.transform(cell_traceB) #testing on A
        pos_test_score_shuffB, pos_test_err_shuffB, dis_mean_shuffB, dis_median_shuffB = pos_score(cebra_loc_train_shuffA, cebra_loc_test_shuffB, pos_trainA, posB)


        #then sanity check use B to decode B

        cell_trainB, cell_testB = hold_out(cell_traceB, percent_to_train)
        pos_trainB, pos_testB = hold_out(posB, percent_to_train)

        cebra_loc_modelB = cebra_loc_model.fit(cell_trainB, pos_trainB)
        cebra_loc_trainB = cebra_loc_modelB.transform(cell_trainB)
        cebra_loc_testB = cebra_loc_modelB.transform(cell_testB)


        pos_test_scoreB, pos_test_errB, dis_meanB, dis_medianB = pos_score(cebra_loc_trainB, cebra_loc_testB, pos_trainB, pos_testB)
        #want pos_test_err


        # For err_allA
        pos_test_scoreA_val = pos_test_scoreA[0] if isinstance(pos_test_scoreA, (list, tuple)) else pos_test_scoreA
        pos_test_errA_val = pos_test_errA[0] if isinstance(pos_test_errA, (list, tuple)) else pos_test_errA
        dis_meanA_val = dis_meanA[0] if isinstance(dis_meanA, (list, tuple)) else dis_meanA
        dis_medianA_val = dis_medianA[0] if isinstance(dis_medianA, (list, tuple)) else dis_medianA

        # Create the tuple
        err_allA = pos_test_scoreA_val, pos_test_errA_val, dis_meanA_val, dis_medianA_val

        # For err_allB_usingA
        pos_test_scoreBwA_val = pos_test_scoreBwA[0] if isinstance(pos_test_scoreBwA, (list, tuple)) else pos_test_scoreBwA
        pos_test_errBwA_val = pos_test_errBwA[0] if isinstance(pos_test_errBwA, (list, tuple)) else pos_test_errBwA
        dis_meanBwA_val = dis_meanBwA[0] if isinstance(dis_meanBwA, (list, tuple)) else dis_meanBwA
        dis_medianBwA_val = dis_medianBwA[0] if isinstance(dis_medianBwA, (list, tuple)) else dis_medianBwA

        # Create the tuple
        err_allB_usingA = pos_test_scoreBwA_val, pos_test_errBwA_val, dis_meanBwA_val, dis_medianBwA_val

        # For err_all_shuffA
        pos_test_score_shuffA_val = pos_test_score_shuffA[0] if isinstance(pos_test_score_shuffA, (list, tuple)) else pos_test_score_shuffA
        pos_test_err_shuffA_val = pos_test_err_shuffA[0] if isinstance(pos_test_err_shuffA, (list, tuple)) else pos_test_err_shuffA
        dis_mean_shuffA_val = dis_mean_shuffA[0] if isinstance(dis_mean_shuffA, (list, tuple)) else dis_mean_shuffA
        dis_median_shuffA_val = dis_median_shuffA[0] if isinstance(dis_median_shuffA, (list, tuple)) else dis_median_shuffA

        # Create the tuple
        err_all_shuffA = pos_test_score_shuffA_val, pos_test_err_shuffA_val, dis_mean_shuffA_val, dis_median_shuffA_val

        # For err_all_shuffB_usingA
        pos_test_score_shuffB_val = pos_test_score_shuffB[0] if isinstance(pos_test_score_shuffB, (list, tuple)) else pos_test_score_shuffB
        pos_test_err_shuffB_val = pos_test_err_shuffB[0] if isinstance(pos_test_err_shuffB, (list, tuple)) else pos_test_err_shuffB
        dis_mean_shuffB_val = dis_mean_shuffB[0] if isinstance(dis_mean_shuffB, (list, tuple)) else dis_mean_shuffB
        dis_median_shuffB_val = dis_median_shuffB[0] if isinstance(dis_median_shuffB, (list, tuple)) else dis_median_shuffB

        # Create the tuple
        err_all_shuffB_usingA = pos_test_score_shuffB_val, pos_test_err_shuffB_val, dis_mean_shuffB_val, dis_median_shuffB_val

        # For err_allB_usingB
        pos_test_scoreB_val = pos_test_scoreB[0] if isinstance(pos_test_scoreB, (list, tuple)) else pos_test_scoreB
        pos_test_errB_val = pos_test_errB[0] if isinstance(pos_test_errB, (list, tuple)) else pos_test_errB
        dis_meanB_val = dis_meanB[0] if isinstance(dis_meanB, (list, tuple)) else dis_meanB
        dis_medianB_val = dis_medianB[0] if isinstance(dis_medianB, (list, tuple)) else dis_medianB

        # Create the tuple
        err_allB_usingB = pos_test_scoreB_val, pos_test_errB_val, dis_meanB_val, dis_medianB_val


    print(np.mean(pos_test_scoreA))
    print(np.mean(pos_test_scoreBwA))
    print(np.mean(pos_test_score_shuffA))
    print(np.mean(pos_test_score_shuffB))
    print(np.mean(pos_test_scoreB))

    del cebra_loc_modelA, cebra_loc_trainA, cebra_loc_testA
    del cebra_loc_testB, shuff_modelA
    del cebra_loc_train_shuffA, cebra_loc_test_shuffA, cebra_loc_test_shuffB
    del cebra_loc_modelB, cebra_loc_trainB

    # Call garbage collector
    gc.collect()

    return err_allA, err_allB_usingA, err_all_shuffA, err_all_shuffB_usingA, err_allB_usingB
