import sys
import argparse
import cebra
from cebra import CEBRA
import cebra.helper as cebra_helper
import numpy as np
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')
from consistency import consistency
import matplotlib.pyplot as plt
import pandas as pd
import joblib as jl
from hold_out import hold_out
from matplotlib.collections import LineCollection


def CSUSconsistency(envA_cell_train, envB_cell_train, envA_eyeblink, envB_eyeblink, dimensions):



    #looks at consistancy between two models
    output_dimension = dimensions #here, we set as a variable for hypothesis testing below.
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

    eyeblink_train_controlA, eyeblink_test_controlA = hold_out(envA_eyeblink, .75)
    cell_train_controlA, cell_test_controlA  = hold_out(envA_cell_train,.75)

    eyeblink_train_controlB, eyeblink_test_controlB = hold_out(envB_eyeblink, .75)
    cell_train_controlB, cell_test_controlB  = hold_out(envB_cell_train,.75)

    if not np.array_equal(eyeblink_test_controlA[:10], eyeblink_test_controlB[:10]):
        # Determine the amount to truncate from the training sets to make the first 10 values the same
        min_length = min(len(eyeblink_test_controlA), len(eyeblink_test_controlB))
        eyeblink_test_controlA, eyeblink_test_controlB = eyeblink_test_controlA[:min_length], eyeblink_test_controlB[:min_length]
        # Truncate traceA and traceB to maintain alignment with the truncated training sets
        cell_test_controlA, cell_test_controlB = cell_test_controlA[:min_length], cell_test_controlB[:min_length]


    model1 = cebra_loc_model.fit(envA_cell_train, envA_eyeblink)
    model1 = model1.transform(cell_test_controlA)

    model2 = cebra_loc_model.fit(envB_cell_train, envB_eyeblink)
    model2 = model2.transform(cell_test_controlB)

    f = consistency([model1, model2])
    print(f)

    return f

def parse_list_argument(arg_value):
    """Converts a comma-separated string to a list of floats or integers."""
    try:
        return [float(item) if '.' in item else int(item) for item in arg_value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Value \"{arg_value}\" is not a valid list of numbers.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run conditional decoding with CEBRA.")
    parser.add_argument("traceA", type=str, help="File path for traceA data.")
    parser.add_argument("traceB", type=str, help="File path for traceB data.")
    parser.add_argument("trainingA", type=str, help="File path for trainingA data.")
    parser.add_argument("trainingB", type=str, help="File path for trainingB data.")
    parser.add_argument("how_many_divisions", type=int, help="Number of divisions for categorizing data.")
    parser.add_argument("pretrial_y_or_n", type=int, choices=[0, 1], help="Pretrial flag (0 or 1).")
    args = parser.parse_args()

    traceA = cebra.load_data(file=args.traceA)  # Adjust 'your_key_here' as necessary
    traceB= cebra.load_data(file=args.traceB)  # Adjust 'your_key_here' as necessary
    trainingA = cebra.load_data(file=args.trainingA)  # Adjust 'your_key_here' as necessary
    trainingB = cebra.load_data(file=args.trainingB)  # Adjust 'your_key_here' as necessary

    # Data preprocessing steps
    trainingA = trainingA[0, :]
    trainingB = trainingB[0, :]
    traceA = np.transpose(traceA)
    traceB = np.transpose(traceB)
    trainingA = trainingA.flatten()
    trainingB = trainingB.flatten()


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
    if how_many_divisions == 2:
        trainingA[(trainingA > 0) & (trainingA <= 5)]  = 1
        trainingA[trainingA > 5] = 2
        trainingA[trainingA == -1] = 0

        trainingB[(trainingB > 0) & (trainingB <= 5)]  = 1
        trainingB[trainingB > 5] = 2
        trainingB[trainingB == -1] = 0
    elif how_many_divisions == 5:
        trainingA[(trainingA > 0) & (trainingA <= 2)]  = 1
        trainingA[(trainingA > 2) & (trainingA <= 4)] = 2
        trainingA[(trainingA > 4) & (trainingA <= 6)] = 3
        trainingA[(trainingA > 6) & (trainingA <= 8)] = 4
        trainingA[trainingA > 8] = 5
        trainingA[trainingA == -1] = 0

        trainingB[(trainingB > 0) & (trainingB <= 2)]  = 1
        trainingB[(trainingB > 2) & (trainingB <= 4)] = 2
        trainingB[(trainingB > 4) & (trainingB <= 6)] = 3
        trainingB[(trainingB > 6) & (trainingB <= 8)] = 4
        trainingB[trainingB > 8] = 5
        trainingB[trainingB == -1] = 0

    dimensions = how_many_divisions + args.pretrial_y_or_n
    CSUSconsistency(traceA, traceB, trainingA, trainingB, dimensions)
