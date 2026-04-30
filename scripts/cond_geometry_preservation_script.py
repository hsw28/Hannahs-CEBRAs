import sys

sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')

import argparse

import cebra
import numpy as np

from cond_geometry_preservation import run_geometry_preservation


parameter_sets = {
    "set0222": {"learning_rate": 0.0035, "min_temperature": 2.33, "max_iterations": 50000, "distance": "euclidean", "temp_mode": "constant"},
    "set0307": {"learning_rate": 0.0025, "min_temperature": 1.5, "max_iterations": 30000, "distance": "cosine", "temp_mode": "constant"},
    "set0313": {"learning_rate": 0.0035, "min_temperature": 1.67, "max_iterations": 20000, "distance": "cosine", "temp_mode": "auto"},
    "set0314": {"learning_rate": 0.0095, "min_temperature": 2.66, "max_iterations": 25000, "distance": "cosine", "temp_mode": "constant"},
    "set0816": {"learning_rate": 0.0095, "min_temperature": 1.67, "max_iterations": 16000, "distance": "cosine", "temp_mode": "auto"},
    "test": {"learning_rate": 0.02, "min_temperature": 0.02, "max_iterations": 100, "distance": "cosine", "temp_mode": "auto"},
}


def bin_csus(labels, how_many_divisions):
    labels = labels.copy()
    if how_many_divisions == 2:
        labels[(labels > 0) & (labels <= 6)] = 1
        labels[labels > 6] = 2
        labels[labels == -1] = 0
    elif how_many_divisions == 5:
        labels[(labels > 0) & (labels <= 2)] = 1
        labels[(labels > 2) & (labels <= 4)] = 2
        labels[(labels > 4) & (labels <= 6)] = 3
        labels[(labels > 6) & (labels <= 8)] = 4
        labels[labels > 8] = 5
        labels[labels == -1] = 0
    elif how_many_divisions == 10:
        labels[labels == -1] = 0
    else:
        raise ValueError("how_many_divisions must be 2, 5, or 10.")
    return labels


def filter_pretrial(trace, labels, pretrial_y_or_n):
    if pretrial_y_or_n == 0:
        return trace[labels > 0], labels[labels > 0]
    return trace[labels != 0], labels[labels != 0]


def filter_paired_training_traces(trace_a, trace_b, labels, pretrial_y_or_n):
    if pretrial_y_or_n == 0:
        mask = labels > 0
    else:
        mask = labels != 0
    return trace_a[mask], trace_b[mask], labels[mask]


parser = argparse.ArgumentParser(description="Quantify CEBRA task-geometry preservation across A and B environments.")
parser.add_argument("traceA1An_An", type=str, help="Path to the traceA1An_An data file.")
parser.add_argument("traceAnB1_An", type=str, help="Path to the traceAnB1_An data file.")
parser.add_argument("traceA1An_A1", type=str, help="Path to the traceA1An_A1 data file.")
parser.add_argument("traceAnB1_B1", type=str, help="Path to the traceAnB1_B1 data file.")
parser.add_argument("CSUSAn", type=str, help="Path to the CSUSAn data file.")
parser.add_argument("CSUSA1", type=str, help="Path to the CSUSA1 data file.")
parser.add_argument("CSUSB1", type=str, help="Path to the CSUSB1 data file.")
parser.add_argument("how_many_divisions", type=int, help="Number of task bins: 2, 5, or 10.")
parser.add_argument("pretrial_y_or_n", type=int, choices=[0, 1], help="Pretrial flag (0 or 1).")
parser.add_argument("--iterations", type=int, default=20, help="Number of independent CEBRA model runs.")
parser.add_argument("--shuffles", type=int, default=1, help="Number of task-bin order shuffles per model run. If >1, their mean is the one rShuff for that run.")
parser.add_argument("--output_dimension", type=int, default=3, help="CEBRA embedding dimensionality.")
parser.add_argument("--parameter_set_name", type=str, default="set0222", help="Name of the parameter set to use.")
parser.add_argument("--output_dir", type=str, default="geometry_preservation_outputs", help="Directory for CSV, NPZ, and PNG outputs.")
parser.add_argument("--rat_id", type=str, default=None, help="Optional rat/session label for saved outputs and plots.")
parser.add_argument("--session_id", type=str, default=None, help="Optional session label for saved outputs.")
parser.add_argument("--random_seed", type=int, default=None, help="Optional random seed for shuffle controls.")
args = parser.parse_args()


traceA1An_An = np.transpose(cebra.load_data(file=args.traceA1An_An))
traceAnB1_An = np.transpose(cebra.load_data(file=args.traceAnB1_An))
traceA1An_A1 = np.transpose(cebra.load_data(file=args.traceA1An_A1))
traceAnB1_B1 = np.transpose(cebra.load_data(file=args.traceAnB1_B1))

CSUSAn = cebra.load_data(file=args.CSUSAn)[0, :].flatten()
CSUSA1 = cebra.load_data(file=args.CSUSA1)[0, :].flatten()
CSUSB1 = cebra.load_data(file=args.CSUSB1)[0, :].flatten()

traceA1An_An, traceAnB1_An, CSUSAn = filter_paired_training_traces(
    traceA1An_An,
    traceAnB1_An,
    CSUSAn,
    args.pretrial_y_or_n,
)
traceA1An_A1, CSUSA1 = filter_pretrial(traceA1An_A1, CSUSA1, args.pretrial_y_or_n)
traceAnB1_B1, CSUSB1 = filter_pretrial(traceAnB1_B1, CSUSB1, args.pretrial_y_or_n)

CSUSAn = bin_csus(CSUSAn, args.how_many_divisions)
CSUSA1 = bin_csus(CSUSA1, args.how_many_divisions)
CSUSB1 = bin_csus(CSUSB1, args.how_many_divisions)

dimensions = args.how_many_divisions + args.pretrial_y_or_n
parameter_set = parameter_sets[args.parameter_set_name]

run_geometry_preservation(
    traceA1An_An,
    traceAnB1_An,
    traceA1An_A1,
    traceAnB1_B1,
    CSUSAn,
    CSUSA1,
    CSUSB1,
    dimensions,
    args.iterations,
    parameter_set,
    parameter_set_name=args.parameter_set_name,
    shuffles=args.shuffles,
    output_dimension=args.output_dimension,
    output_dir=args.output_dir,
    rat_id=args.rat_id,
    session_id=args.session_id,
    random_seed=args.random_seed,
)
