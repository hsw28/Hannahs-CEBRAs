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
from cebra import CEBRA
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection
from CSUS_score import CSUS_score
from hold_out import hold_out
import gc


#grid search

#how to run:
    #conda activate cebra
    #python cond_decoding_AvsB_grid.py envA_cell_train envB_cell_train envA_eyeblink envB_eyeblink how_many_divisions pretrial_y_or_n

#pretrial_y_or_n: 0 for only cs us, 1 for cs us pretrial
#how many divisions you wanted-- for ex,
#pretrial_y_or_n = 1
    # how_many_divisions = 2 will just split between cs and us
                        #= 10 will split CS and US each into 5

#Example
    # python cond_decoding_AvsB_grid.py ./traceAnB1_An.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat 2 0 --learning_rate 1e-4,5e-4 --min_temperature 0.1,0.3 --max_iterations 5000,10000


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
    parser.add_argument("trainingA", type=str, help="File path for trainingA data.")
    parser.add_argument("trainingB", type=str, help="File path for trainingB data.")
    parser.add_argument("how_many_divisions", type=int, help="Number of divisions for categorizing data.")
    parser.add_argument("pretrial_y_or_n", type=int, choices=[0, 1], help="Pretrial flag (0 or 1).")
    parser.add_argument("--learning_rate", type=parse_list_argument, default=[8.6e-4], help="Comma-separated learning rates.")
    parser.add_argument("--min_temperature", type=parse_list_argument, default=[0.2], help="Comma-separated minimum temperatures.")
    parser.add_argument("--max_iterations", type=parse_list_argument, default=[8000], help="Comma-separated max iterations.")
    args = parser.parse_args()

    traceA = cebra.load_data(file=args.traceA)  # Adjust 'your_key_here' as necessary
    traceB= cebra.load_data(file=args.traceB)  # Adjust 'your_key_here' as necessary
    trainingA = cebra.load_data(file=args.trainingA)  # Adjust 'your_key_here' as necessary
    trainingB = cebra.load_data(file=args.trainingB)  # Adjust 'your_key_here' as necessary

    # Data preprocessing steps
    trainingA = trainingA[0, :]
    trainingB = trainingB[0, :]
    trainingA = trainingA.flatten()
    trainingB = trainingB.flatten()

    traceA = np.transpose(traceA)
    traceB = np.transpose(traceB)


    # Logic to divide data based on 'divisions' and 'pretrial'
    if args.pretrial_y_or_n == 0:
        traceA = traceA[trainingA > 0]
        trainingA = trainingA[trainingA > 0]

        traceB = traceB[trainingB > 0]
        trainingB = trainingB[trainingB > 0]
    else:
        traceA = traceA[trainingA != 0]
        trainingA = trainingA[trainingA != 0]

        traceB = traceB[trainingB != 0]
        trainingB = trainingB[trainingB != 0]


    how_many_divisions = args.how_many_divisions
#    if how_many_divisions == 2:
#        trainingA[(trainingA > 0) & (trainingA <= 5)]  = 1
#        trainingA[trainingA > 5] = 2
#        trainingA[trainingA == -1] = 0

#        trainingB[(trainingB > 0) & (trainingB <= 5)]  = 1
#        trainingB[trainingB > 5] = 2
#        trainingB[trainingB == -1] = 0

    if how_many_divisions == 2:
        trainingA[(trainingA > 0) & (trainingA <= 6)]  = 1
        trainingA[trainingA > 6] = 2
        trainingA[trainingA == -1] = 0

        trainingB[(trainingB > 0) & (trainingB <= 6)]  = 1
        trainingB[trainingB > 6] = 2
        trainingB[trainingB == -1] = 0

    elif how_many_divisions == 5:
        trainingA[(trainingA > 0) & (trainingA <= 2)]  = 1
        trainingA[(trainingA > 2) & (trainingA <= 4)] = 2
        trainingA[(trainingA > 4) & (trainingA <= 6)] = 3
        trainingA[(trainingA > 6) & (trainingA <= 8)] = 4
        trainingA[trainingA > 8] = 5
        trainingA[trainingA == -1] = 0

        training[(trainingB > 0) & (trainingB <= 2)]  = 1
        trainingB[(trainingB > 2) & (trainingB <= 4)] = 2
        trainingB[(trainingB > 4) & (trainingB <= 6)] = 3
        trainingB[(trainingB > 6) & (trainingB <= 8)] = 4
        trainingB[trainingB > 8] = 5
        trainingB[trainingB == -1] = 0

    dimensions = how_many_divisions + args.pretrial_y_or_n

    # Run the grid search
    results = cond_decoding_AvsB_grid_cebra(
        traceA, traceB, trainingA, trainingB,
        args.learning_rate,
        args.min_temperature,
        args.max_iterations,
        dimensions
    )


    #print(results)


def cond_decoding_AvsB_grid_cebra(envA_cell_train, envB_cell_train, envA_eyeblink, envB_eyeblink, learning_rates, min_temperatures, max_iterations_list, dimensions):
    results = []
    for lr, temp, max_iter in product(learning_rates, min_temperatures, max_iterations_list):
        #print({'learning_rate': lr, 'min_temperature': temp, 'max_iterations': max_iter})
        # Setup the CEBRA model with the current set of parameters
        cebra_loc_model = CEBRA(
            learning_rate=lr,
            max_iterations=max_iter,
            model_architecture='offset10-model',
            batch_size=512,
            temperature_mode= 'constant',
            #temperature_mode='auto',
            #min_temperature=temp,
            temperature=temp,
            output_dimension=dimensions,
            distance='cosine',
            conditional='time_delta',
            device='cuda_if_available',
            num_hidden_units=32,
            time_offsets=1,
            verbose=False
        )

        fract_control_all = []
        fract_test_all = []
        loss_all = []

        # Loop to run the batch of code 50 times
        for i in range(3):



              #test control environment

              ######### use this to test in own environment
              eyeblink_train_control, eyeblink_test_control = hold_out(envA_eyeblink, .75)
              cell_train_control, cell_test_control  = hold_out(envA_cell_train,.75)

              #run the model
              cebra_loc_modelpos = cebra_loc_model.fit(cell_train_control, eyeblink_train_control)
              loss_all.append(cebra_loc_model.state_dict_['loss'][-1])
              #determine model fit
              cebra_loc_train22 = cebra_loc_modelpos.transform(cell_train_control)
              cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test_control)


              #find fraction correct
              fract_controlA = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)


              #test with using A to decode B


              #if i want to fit B using fulling training, but i think i want to do it with held out
              '''
              cebra_loc_modelpos_full = cebra_loc_model.fit(envA_cell_train, envA_eyeblink)
              loss_all.append(cebra_loc_model.state_dict_['loss'][-1])
              #determine model fit
              cebra_loc_train22 = cebra_loc_modelpos.transform(envA_cell_train)
              cebra_loc_test22 = cebra_loc_modelpos_full.transform(cell_test)
              #find fraction correct
              fract_testB = CSUS_score(cebra_loc_train22, cebra_loc_test22, envA_eyeblink, eyeblink_test_control)
              '''


              #determine model fit
              cebra_loc_test22 = cebra_loc_modelpos.transform(envB_cell_train)
              #find fraction correct
              fract_testB = CSUS_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, envB_eyeblink)

              fract_controlA = round(fract_controlA,3)
              fract_testB = round(fract_testB,3)

              fract_control_all.append(fract_controlA)
              fract_test_all.append(fract_testB)


              del cebra_loc_modelpos, cebra_loc_train22, cebra_loc_test22
              gc.collect()

              #print((fract_control_all))
              #print((fract_test_all))

        # Calculate mean of all fractions
        mean_control = np.mean(fract_control_all)
        mean_test = np.mean(fract_test_all)  # Corrected to use fract_test_all
        mean_loss = np.mean(loss_all)
        std_loss = np.std(loss_all)


        # Round the mean values
        mean_control = round(mean_control, 3)
        mean_test = round(mean_test, 3)
        mean_loss = round(mean_loss,3)
        std_loss = round(std_loss,3)

        # Append the correctly calculated means to the results
        results.append({
            'learn_rate': lr,
            'min_temp': temp,
            'max_it': max_iter,
        #    'fract_control': fract_control_all,
        #    'fract_test': fract_test_all,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'mean_control': mean_control,  # Correctly calculated mean
            'mean_test': mean_test         # Correctly calculated mean
        })

        print(results)
    return results

if __name__ == "__main__":
    main()
