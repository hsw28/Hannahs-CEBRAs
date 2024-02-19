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
from pos_score import pos_score
from hold_out import hold_out
import gc


#grid search

#how to run:
    #conda activate cebra
    #python cond_decoding_AvsB_script.py traceA_file traceB_file trainingA_file trainingB_file

#pretrial_y_or_n: 0 for only cs us, 1 for cs us pretrial
#how many divisions you wanted-- for ex,
#pretrial_y_or_n = 1
    # how_many_divisions = 2 will just split between cs and us
                        #= 10 will split CS and US each into 5

#Example
    # python cond_decoding_script.py traceA.npy traceB.npy trainingA.npy trainingB.npy --learning_rate 1e-4,5e-4 --min_temperature 0.1,0.3 --max_iterations 5000,10000


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
    parser.add_argument("--learning_rate", type=parse_list_argument, default=[8.6e-4], help="Comma-separated learning rates.")
    parser.add_argument("--min_temperature", type=parse_list_argument, default=[0.2], help="Comma-separated minimum temperatures.")
    parser.add_argument("--max_iterations", type=parse_list_argument, default=[8000], help="Comma-separated max iterations.")
    args = parser.parse_args()

    traceA = cebra.load_data(file=args.traceA)  # Adjust 'your_key_here' as necessary
    traceB= cebra.load_data(file=args.traceB)  # Adjust 'your_key_here' as necessary
    trainingA = cebra.load_data(file=args.trainingA)  # Adjust 'your_key_here' as necessary
    trainingB = cebra.load_data(file=args.trainingB)  # Adjust 'your_key_here' as necessary

    # Data preprocessing steps
    traceA = np.transpose(traceA)
    traceB = np.transpose(traceB)

    trainingA = trainingA[:,1:]
    trainingB = trainingB[:,1:]


    # Run the grid search
    results = pos_decoding_AvsB_grid_cebra(
        traceA, trainingA, traceB, trainingB,
        args.learning_rate,
        args.min_temperature,
        args.max_iterations,
    )


    #print(results)


def pos_decoding_AvsB_grid_cebra(envA_cell_train, posA, envB_cell_train, posB, learning_rates, min_temperatures, max_iterations_list):
    results = []
    for lr, temp, max_iter in product(learning_rates, min_temperatures, max_iterations_list):
        #print({'learning_rate': lr, 'min_temperature': temp, 'max_iterations': max_iter})
        # Setup the CEBRA model with the current set of parameters
        cebra_loc_model = CEBRA(
            learning_rate=lr,
            min_temperature=temp,
            max_iterations=max_iter,
            model_architecture='offset10-model',
            batch_size=512,
            temperature_mode='auto',
            output_dimension=3,
            distance='cosine',
            conditional='time_delta',
            device='cuda_if_available',
            num_hidden_units=32,
            time_offsets=1,
            verbose=True
        )

        fract_control_all = []
        fract_test_all = []
        med_control_all = []
        med_test_all = []

        # Loop to run the batch of code 50 times
        for i in range(3):



              #test control environment

              ######### use this to test in own environment
              eyeblink_train_control, eyeblink_test_control = hold_out(posA, 1)
              eyeblink_test_control = eyeblink_train_control
              cell_train_control, cell_test_control  = hold_out(envA_cell_train, 1)
              cell_test_control = cell_train_control

              #run the model
              cebra_loc_modelpos = cebra_loc_model.fit(cell_train_control, eyeblink_train_control)
              #determine model fit
              cebra_loc_train22 = cebra_loc_modelpos.transform(cell_train_control)
              cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test_control)


              #find fraction correct
              pos_test_score_train, pos_test_err_train, dis_mean_train, dis_median_train = pos_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)



              #test with using A to decode B
              cell_test = envB_cell_train
              eyeblink_test_control = posB

              #determine model fit
              cebra_loc_test22 = cebra_loc_modelpos.transform(cell_test)
              #find fraction correct
              pos_test_score_test, pos_test_err_test, dis_mean_test, dis_median_test = pos_score(cebra_loc_train22, cebra_loc_test22, eyeblink_train_control, eyeblink_test_control)


              fract_control_all.append(pos_test_err_train)
              fract_test_all.append(pos_test_score_test)
              med_control_all.append(dis_median_train)
              med_test_all.append(dis_median_test)


              del cebra_loc_modelpos, cebra_loc_train22, cebra_loc_test22
              gc.collect()

              #print((fract_control_all))
              #print((fract_test_all))

        # Calculate mean of all fractions
        fract_control_all = np.mean(fract_control_all)
        fract_test_all = np.mean(fract_test_all)  # Corrected to use fract_test_all
        med_control_all = np.mean(med_control_all)
        med_test_all = np.mean(med_test_all)



        # Append the correctly calculated means to the results
        results.append({
            'learn_rate': lr,
            'min_temp': temp,
            'max_it': max_iter,
            'mean_control': fract_control_all,
            'mean_test': fract_test_all,
            'med_control': med_control_all,  # Correctly calculated mean
            'mead_test': med_test_all       # Correctly calculated mean
        })

        print(results)
    return results

if __name__ == "__main__":
    main()
