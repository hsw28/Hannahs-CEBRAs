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
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection
from pos_score import pos_score
from hold_out import hold_out
import gc
from smoothpos import smoothpos
from ca_velocity import ca_velocity


#grid search

#how to run:
    #conda activate cebra
    #python cond_decoding_AvsB_script.py traceA_file traceB_file PosA_file PosB_file

#pretrial_y_or_n: 0 for only cs us, 1 for cs us pretrial
#how many divisions you wanted-- for ex,
#pretrial_y_or_n = 1
    # how_many_divisions = 2 will just split between cs and us
                        #= 10 will split CS and US each into 5

#Example
    # python cond_decoding_AvsB_script.py traceA.npy traceB.npy PosA.npy PosB.npy --learning_rate 1e-4,5e-4 --min_temperature 0.1,0.3 --max_iterations 5000,10000


def parse_list_argument(arg_value):
    """Converts a comma-separated string to a list of floats or integers."""
    try:
        return [float(item) if '.' in item else int(item) for item in arg_value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Value \"{arg_value}\" is not a valid list of numbers.")

def main():
    parser = argparse.ArgumentParser(description="Run conditional decoding with CEBRA.")
    parser.add_argument("traceA", type=str, help="File path for traceA data.")
    parser.add_argument("traceB", type=str, help="File path for traceB data.")
    parser.add_argument("PosA", type=str, help="File path for PosA data.")
    parser.add_argument("PosB", type=str, help="File path for PosB data.")
    parser.add_argument("--learning_rate", type=parse_list_argument, default=[8.6e-4], help="Comma-separated learning rates.")
    parser.add_argument("--min_temperature", type=parse_list_argument, default=[0.2], help="Comma-separated minimum temperatures.")
    parser.add_argument("--max_iterations", type=parse_list_argument, default=[8000], help="Comma-separated max iterations.")
    args = parser.parse_args()

    traceA = cebra.load_data(file=args.traceA)  # Adjust 'your_key_here' as necessary
    traceB= cebra.load_data(file=args.traceB)  # Adjust 'your_key_here' as necessary
    PosA = cebra.load_data(file=args.PosA)  # Adjust 'your_key_here' as necessary
    PosB = cebra.load_data(file=args.PosB)  # Adjust 'your_key_here' as necessary

    # Data preprocessing steps
    traceA = np.transpose(traceA)
    traceB = np.transpose(traceB)

    PosA = smoothpos(PosA)
    PosB = smoothpos(PosB)

    PosA = PosA[:,1:]
    PosA = PosA[::2]
    if len(PosA) > len(traceA):
        PosA = PosA[:len(traceA)]

    PosB = PosB[:,1:]
    PosB = PosB[::2]
    if len(PosB) > len(traceB):
        PosB = PosB[:len(traceB)]


    vel_A = ca_velocity(PosA)
    vel_B = ca_velocity(PosB)
    high_vel_indices_A = np.where(vel_A >= 4)[0]
    high_vel_indices_B = np.where(vel_B >= 4)[0]
    if high_vel_indices_A.size > 0 and high_vel_indices_A[-1] + 1 < len(PosA):
        high_vel_indices_A = high_vel_indices_A + 1
    if high_vel_indices_B.size > 0 and high_vel_indices_B[-1] + 1 < len(PosB):
        high_vel_indices_B = high_vel_indices_B + 1
    PosA = PosA[high_vel_indices_A]
    PosB = PosB[high_vel_indices_B]
    traceA = traceA[high_vel_indices_A]
    traceB = traceB[high_vel_indices_B]



    # Run the grid search
    results = pos_decoding_AvsB_grid_cebra(
        traceA, PosA, traceB, PosB,
        args.learning_rate,
        args.min_temperature,
        args.max_iterations,
    )


    #print(results)


def pos_decoding_AvsB_grid_cebra(envA_cell_train, PosA, envB_cell_train, PosB, learning_rates, min_temperatures, max_iterations_list):
    results = []
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
            #temperature=temp,
            output_dimension=3,
            distance='cosine',
            conditional='time_delta',
            device='cuda_if_available',
            num_hidden_units=32,
            time_offsets=1,
            verbose=False
        )

        Pos_err_shuff_all = []
        Pos_err_train_all = []
        Pos_err_test_all = []
        Pos_r2_score_train_all = []
        Pos_r2_score_test_all = []
        med_control_all = []
        med_test_all = []
        med_shuff_all = []



        # Loop to run the batch of code 50 times
        for i in range(3):



              #test control environment

              ######### use this to test in own environment
              eyeblink_train_control, eyeblink_test_control = hold_out(PosA, .8)
              cell_train_control, cell_test_control  = hold_out(envA_cell_train, .8)


              #run the model
              cebra_loc_modelPos = cebra_loc_model.fit(cell_train_control, eyeblink_train_control)
              #determine model fit
              cebra_loc_train22 = cebra_loc_modelPos.transform(cell_train_control)
              cebra_loc_test22 = cebra_loc_modelPos.transform(cell_test_control)


              #find fraction correct

              Pos_test_score_train, Pos_test_err_train, dis_mean_train, dis_median_train = pos_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)
              #pos_test_score: The R² score for both position predictions
              #It represents the proportion of variance in the dependent variable that is predictable from the independent variables.
              #pos_test_err: The median absolute error between the predicted positions and the true positions.
              #This provides a robust measure of the error magnitude.

              #test with using A to decode B
              cell_test = envB_cell_train
              eyeblink_test_control = PosB
              #determine model fit
              cebra_loc_test22 = cebra_loc_modelPos.transform(cell_test)
              #find fraction correct
              Pos_test_score_test, Pos_test_err_test, dis_mean_test, dis_median_test = pos_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)
              #r2, Knn_pos_err, distance_mean, distance_median


              #now shuffled
              indices = np.random.permutation(PosA.shape[0])  # Get a permutation of the row indices
              posAn_shuffled = PosA[indices, :] #apply and shuffle

              eyeblink_train_control, eyeblink_test_control = hold_out(posAn_shuffled, .8)
              cell_train_control, cell_test_control  = hold_out(envA_cell_train, .8)

              cell_test = envB_cell_train
              eyeblink_test_control = PosB


              cebra_loc_modelPos = cebra_loc_model.fit(cell_train_control, eyeblink_train_control)

              cebra_loc_train22 = cebra_loc_modelPos.transform(cell_train_control)
              cebra_loc_test22 = cebra_loc_modelPos.transform(cell_test)

              Pos_test_score_test_shuff, Pos_test_err_test_shuff, dis_mean_test_shuff, dis_median_test_shuff = pos_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)



              Pos_err_shuff_all.append(Pos_test_err_test_shuff)
              med_shuff_all.append(dis_median_test_shuff)

              Pos_err_train_all.append(Pos_test_err_train)
              Pos_err_test_all.append(Pos_test_err_test)

              Pos_r2_score_train_all.append(Pos_test_score_train)
              Pos_r2_score_test_all.append(Pos_test_score_test)

              med_control_all.append(dis_median_train)
              med_test_all.append(dis_median_test)


              del cebra_loc_modelPos, cebra_loc_train22, cebra_loc_test22
              gc.collect()

              #print((Pos_err_train_all))
              #print((Pos_r2_score_test_all))

        # Calculate mean of all fractions

        Pos_err_shuff_all = np.array(Pos_err_shuff_all)
        Pos_err_test_all = np.array(Pos_err_test_all)
        med_shuff_all = np.array(med_shuff_all)
        med_test_all = np.array(med_test_all)

        shuff_dif_all = np.mean(Pos_err_shuff_all - Pos_err_test_all)
        shuff_med_all = np.mean(med_test_all-med_shuff_all)

        Pos_err_train_all = np.mean(Pos_err_train_all)
        Pos_err_test_all = np.mean(Pos_err_test_all)

        Pos_r2_score_train_all = np.mean(Pos_r2_score_train_all)
        Pos_r2_score_test_all = np.mean(Pos_r2_score_test_all)

        med_control_all = np.mean(med_control_all)
        med_test_all = np.mean(med_test_all)





        # Append the correctly calculated means to the results
        results.append({
            'learn_rate': lr,
            'min_temp': temp,
            'max_it': max_iter,
            'KNN_err_train': Pos_err_train_all,
            'KNN_err_test': Pos_err_test_all,
            'train_r2': Pos_r2_score_train_all,
            'test_r2': Pos_r2_score_test_all,
            'med_control': med_control_all,  # Correctly calculated mean
            'mead_test': med_test_all,       # Correctly calculated mean
            'shuff_minus_not': shuff_dif_all,
            'shuf_med': shuff_med_all
        })

        Pos_err_shuff_all = []
        Pos_err_train_all = []
        Pos_err_test_all = []
        Pos_r2_score_train_all = []
        Pos_r2_score_test_all = []
        med_control_all = []
        med_test_all = []
        med_shuff_all = []

        print(results)
    return results

if __name__ == "__main__":
    main()
