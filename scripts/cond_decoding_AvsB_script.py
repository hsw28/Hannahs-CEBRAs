import argparse
from cebra import CEBRA
import numpy as np
from cond_decoding_AvsB import cond_decoding_AvsB

#how to run:
    #conda activate cebra
    #python cond_decoding_AvsB_script.py traceA_file traceB_file trainingA_file trainingB_file how_many_divisions pretrial_y_or_n

#pretrial_y_or_n: 0 for only cs us, 1 for cs us pretrial
#how many divisions you wanted-- for ex,
#pretrial_y_or_n = 1
    # how_many_divisions = 2 will just split between cs and us
                        #= 10 will split CS and US each into 5

# Setup argument parser
parser = argparse.ArgumentParser(description="Run conditional decoding with CEBRA.")
parser.add_argument("traceA", type=str, help="File path for traceA data.")
parser.add_argument("traceB", type=str, help="File path for traceB data.")
parser.add_argument("trainingA", type=str, help="File path for trainingA data.")
parser.add_argument("trainingB", type=str, help="File path for trainingB data.")
parser.add_argument("divisions", type=int, help="Number of divisions for categorizing data.")
parser.add_argument("pretrial", type=int, choices=[0, 1], help="Pretrial flag (0 or 1).")

# Parse arguments
args = parser.parse_args()

# Load and process data using the arguments
cebra = CEBRA()

traceA = cebra.load_data(args.traceA)
traceA = np.transpose(traceA)

traceB = cebra.load_data(args.traceB)
traceB = np.transpose(traceB)

trainingA = cebra.load_data(args.trainingA)
trainingA = trainingA.flatten()

trainingB = cebra.load_data(args.trainingB)
trainingB = trainingB.flatten()

# Logic to divide data based on 'divisions' and 'pretrial'
if args.pretrial_y_or_n == 0:
    trainingA = trainingA[trainingA > 0]
else:
    # Assuming '!>' was intended to mean 'not greater than' or '<='
    trainingA = trainingA[trainingA != 0]

how_many_divisions = args.how_many_divisions
if divisions == 2:
    trainingA[trainingA <= 5] = 1
    trainingA[trainingA > 5] = 2
elif divisions == 5:
    trainingA[trainingA <= 2] = 1
    trainingA[(trainingA > 2) & (trainingA <= 4)] = 2
    trainingA[(trainingA > 4) & (trainingA <= 6)] = 3
    trainingA[(trainingA > 6) & (trainingA <= 8)] = 4
    trainingA[trainingA > 8] = 5

# Running the conditional decoding function
fract_control_all, fract_test_all = cond_decoding_AvsB(traceA, trainingA, traceB, trainingB)
