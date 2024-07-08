import sys
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')
import argparse
import cebra
from cebra import CEBRA
import cebra.helper as cebra_helper
import numpy as np
from pos_compare_iterations import pos_compare_iterations
from smoothpos import smoothpos
from ca_velocity import ca_velocity

#for using with slurm to run over a bunch of iterations
# python pos_compare_iterations_script.py traceA1An_An traceAnB1_An traceA1An_A1 traceAnB1_B1 PosAn PosA1 PosB1 --iterations 10 --parameter_set_name set0307
# python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/pos_compare_iterations_script.py ./traceA1An_An.mat ./traceAnB1_An.mat ./traceA1An_A1.mat ./traceAnB1_B1.mat ./PosAn.mat ./PosA1.mat ./PosB1.mat --iterations 10 --parameter_set_name set0222

# Define parameter sets
parameter_sets = {
    "set0222": {"learning_rate": 0.000055, "min_temperature": 0.000000001, "max_iterations": 25000},
    #"set0307": {"learning_rate": 0.0006625, "min_temperature": 1.5, "max_iterations": 8000},
    #"set0307": {"learning_rate": 0.00055, "min_temperature": 1.5, "max_iterations": 8000},
    "set0307": {"learning_rate": 0.00055, "min_temperature": 1.5, "max_iterations": 12000},
    #"set0313": {"learning_rate": 0.001, "min_temperature": 0.4, "max_iterations": 10000},
    #"set0313": {"learning_rate": 0.0006625, "min_temperature": 0.000000001, "max_iterations": 22500},
    #"set0313": {"learning_rate": 0.00055, "min_temperature": 0.25, "max_iterations": 20000},
    "set0313": {"learning_rate": 0.00055, "min_temperature": 1, "max_iterations": 30000},
    "set0313b": {"learning_rate": 0.00055, "min_temperature": 0.1, "max_iterations": 25000},

    "set0313c": {"learning_rate": 0.00055, "min_temperature": 0.75, "max_iterations": 20000},
    "set0313d": {"learning_rate": 0.0055, "min_temperature": 1, "max_iterations": 30000},
    #"set0314": {"learning_rate": 0.001, "min_temperature": 0.05, "max_iterations": 30000},
    "set0314": {"learning_rate": 0.001, "min_temperature": 0.000000001, "max_iterations": 30000},
    "set0816": {"learning_rate": 0.001, "min_temperature": 0.15, "max_iterations": 18000}
}


# Setup argparse for command line arguments
parser = argparse.ArgumentParser(description="Run decoding with CEBRA.")
parser.add_argument("traceA1An_An", type=str, help="Path to the traceA1An_An data file.")
parser.add_argument("traceAnB1_An", type=str, help="Path to the traceAnB1_An data file.")
parser.add_argument("traceA1An_A1", type=str, help="Path to the traceA1An_A1 data file.")
parser.add_argument("traceAnB1_B1", type=str, help="Path to the traceAnB1_B1 data file.")
parser.add_argument("PosAn", type=str, help="Path to the PosAn data file.")
parser.add_argument("PosA1", type=str, help="Path to the PosA1 data file.")
parser.add_argument("PosB1", type=str, help="Path to the PosB1 data file.")
parser.add_argument("--iterations", type=int, default=100, help="Number of iterations to run.")
parser.add_argument("--parameter_set_name", type=str, default="set0222", help="Name of the parameter set to use.")

# Parse arguments
args = parser.parse_args()

traceA1An_An = cebra.load_data(file=args.traceA1An_An)  # Adjust 'your_key_here' as necessary
traceAnB1_An = cebra.load_data(file=args.traceAnB1_An)  # Adjust 'your_key_here' as necessary
traceA1An_A1 = cebra.load_data(file=args.traceA1An_A1)  # Adjust 'your_key_here' as necessary
traceAnB1_B1 = cebra.load_data(file=args.traceAnB1_B1)  # Adjust 'your_key_here' as necessary
PosAn = cebra.load_data(file=args.PosAn)  # Adjust 'your_key_here' as necessary
PosA1 = cebra.load_data(file=args.PosA1)  # Adjust 'your_key_here' as necessary
PosB1 = cebra.load_data(file=args.PosB1)  # Adjust 'your_key_here' as necessary

traceA1An_An = np.transpose(traceA1An_An)
traceAnB1_An = np.transpose(traceAnB1_An)
traceA1An_A1 = np.transpose(traceA1An_A1)
traceAnB1_B1 = np.transpose(traceAnB1_B1)

PosA1 = smoothpos(PosA1)
PosAn = smoothpos(PosAn)
PosB1 = smoothpos(PosB1)

#get every other point and check length
PosA1 = PosA1[:,1:]
PosA1 = PosA1[::2]
if len(PosA1) > len(traceA1An_A1):
    PosA1 = PosA1[:len(traceA1An_A1)]

PosAn = PosAn[:,1:]
PosAn = PosAn[::2]
if len(PosAn) > len(traceA1An_An):
    PosAn = PosAn[:len(traceA1An_An)]

PosB1 = PosB1[:,1:]
PosB1 = PosB1[::2]
if len(PosB1) > len(traceAnB1_B1):
    PosB1 = PosB1[:len(traceAnB1_B1)]

vel_A1 = ca_velocity(PosA1)
vel_An = ca_velocity(PosAn)
vel_B1 = ca_velocity(PosB1)

#print(PosA1.shape)
#print(vel_A1.shape)

high_vel_indices_A1 = np.where(vel_A1 >= 4)[0]
high_vel_indices_An = np.where(vel_An >= 4)[0]
high_vel_indices_B1 = np.where(vel_B1 >= 4)[0]

if high_vel_indices_A1.size > 0 and high_vel_indices_A1[-1] + 1 < len(PosA1):
    high_vel_indices_A1 = high_vel_indices_A1 + 1
if high_vel_indices_An.size > 0 and high_vel_indices_An[-1] + 1 < len(PosAn):
    high_vel_indices_An = high_vel_indices_An + 1
if high_vel_indices_B1.size > 0 and high_vel_indices_B1[-1] + 1 < len(PosB1):
    high_vel_indices_B1 = high_vel_indices_B1 + 1

PosA1 = PosA1[high_vel_indices_A1]
PosAn = PosAn[high_vel_indices_An]
PosB1 = PosB1[high_vel_indices_B1]

traceA1An_A1 = traceA1An_A1[high_vel_indices_A1]
traceA1An_An = traceA1An_An[high_vel_indices_An]
traceAnB1_An = traceAnB1_An[high_vel_indices_An]
traceAnB1_B1 = traceAnB1_B1[high_vel_indices_B1]

parameter_set = parameter_sets[args.parameter_set_name]
pos_compare_iterations(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, PosAn, PosA1, PosB1, args.iterations, parameter_set)
