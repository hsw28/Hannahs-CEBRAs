import sys

sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs')
sys.path.append('/home/hsw967/Programming/Hannahs-CEBRAs/scripts')

import argparse

import cebra
import numpy as np

from cond_geometry_preservation import run_geometry_preservation


parameter_sets = {
    "set0222": {"learning_rate": 0.0055, "min_temperature": 1.5, "max_iterations": 77000, "distance": "euclidean", "temp_mode": "constant"},
    "set0222b": {"learning_rate": 0.0055, "min_temperature": 1.5, "max_iterations": 77000, "distance": "euclidean", "temp_mode": "constant"},

    "set0307": {"learning_rate": 0.0025, "min_temperature": 4, "max_iterations": 70000, "distance": "euclidean", "temp_mode": "constant"},
    "set0307_dayb2": {"learning_rate": 0.002, "min_temperature": 2.33, "max_iterations": 80000, "distance": "cosine", "temp_mode": "constant"},
    "set0313": {"learning_rate": 0.0035, "min_temperature": 1.67, "max_iterations": 20000, "distance": "cosine", "temp_mode": "auto"},
    "set0314": {"learning_rate": 0.0075, "min_temperature": 1.67, "max_iterations": 18000, "distance": "euclidean", "temp_mode": "constant"},
    "set0816": {"learning_rate": 0.0095, "min_temperature": 2.66, "max_iterations": 25000, "distance": "cosine", "temp_mode": "constant"},
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


def filter_pretrial_optional_ids(trace, labels, trial_ids, pretrial_y_or_n):
    if pretrial_y_or_n == 0:
        mask = labels > 0
    else:
        mask = labels != 0
    if trial_ids is None:
        return trace[mask], labels[mask], None
    if len(trial_ids) != len(labels):
        raise ValueError(f"trial_ids length {len(trial_ids)} does not match label length {len(labels)}.")
    return trace[mask], labels[mask], np.asarray(trial_ids)[mask]


def filter_paired_training_traces(trace_a, trace_b, labels, pretrial_y_or_n):
    if pretrial_y_or_n == 0:
        mask = labels > 0
    else:
        mask = labels != 0
    return trace_a[mask], trace_b[mask], labels[mask]


parser = argparse.ArgumentParser(description="Quantify CEBRA task-geometry preservation between separately trained A(n) and B(1) embeddings.")
parser.add_argument("traceA1An_An", nargs="?", type=str, help="Legacy path to the traceA1An_An data file.")
parser.add_argument("traceAnB1_An", nargs="?", type=str, help="Legacy path to the traceAnB1_An data file.")
parser.add_argument("traceA1An_A1", nargs="?", type=str, help="Legacy path to the traceA1An_A1 data file.")
parser.add_argument("traceAnB1_B1", nargs="?", type=str, help="Legacy path to the traceAnB1_B1 data file.")
parser.add_argument("CSUSAn", nargs="?", type=str, help="Legacy path to the CSUSAn data file.")
parser.add_argument("CSUSA1", nargs="?", type=str, help="Legacy path to the CSUSA1 data file.")
parser.add_argument("CSUSB1", nargs="?", type=str, help="Legacy path to the CSUSB1 data file.")
parser.add_argument("how_many_divisions", nargs="?", type=int, help="Number of task bins: 2, 5, or 10.")
parser.add_argument("pretrial_y_or_n", nargs="?", type=int, choices=[0, 1], help="Pretrial flag (0 or 1).")
parser.add_argument("--iterations", type=int, default=20, help="Number of independent CEBRA model runs.")
parser.add_argument("--shuffles", type=int, default=1, help="Number of task-bin order shuffles per model run. Use 0 to skip geometry shuffle controls and only save embeddings.")
parser.add_argument("--output_dimension", type=int, default=3, help="CEBRA embedding dimensionality.")
parser.add_argument("--parameter_set_name", type=str, default="set0222", help="Name of the parameter set to use.")
parser.add_argument("--output_dir", type=str, default="geometry_preservation_outputs", help="Directory for CSV, NPZ, and PNG outputs.")
parser.add_argument("--rat_id", type=str, default=None, help="Optional rat/session label for saved outputs and plots.")
parser.add_argument("--session_id", type=str, default=None, help="Optional session label for saved outputs.")
parser.add_argument("--random_seed", type=int, default=None, help="Optional random seed for shuffle controls.")
parser.add_argument("--save_branch", choices=["both", "A", "B"], default="both", help="Which CEBRA branch to fit/save. Use A to save only A(n)-trained embeddings, B to save only B(1)-trained embeddings, or both to compare A(n) and B(1) geometry.")
parser.add_argument("--trial_ids_A1", type=str, default=None, help="Optional trial IDs for traceA1An_A1 samples.")
parser.add_argument("--trial_ids_B1", type=str, default=None, help="Optional trial IDs for traceAnB1_B1 samples.")
parser.add_argument("--traceAn_full", type=str, default=None, help="Optional full-population A(n) trace file. When provided, this replaces traceAnB1_An for geometry preservation.")
parser.add_argument("--traceB1_full", type=str, default=None, help="Optional full-population B(1) trace file. When provided, this replaces traceAnB1_B1 for geometry preservation.")
parser.add_argument("--resume_checkpoint", type=str, default=None, help="Optional *_checkpoint.npz file from a timed-out run. Completed runs are loaded and remaining runs are appended.")
parser.add_argument("--labelsAn", type=str, default=None, help="Full-population mode path to the A(n) CS-US label file.")
parser.add_argument("--labelsB1", type=str, default=None, help="Full-population mode path to the B(1) CS-US label file.")
parser.add_argument("--labelsA1", type=str, default=None, help="Optional legacy A(1) CS-US label file. Not used for full-population geometry.")
parser.add_argument("--task_bins", type=int, choices=[2, 5, 10], default=None, help="Number of task bins for clean full-population CLI mode.")
parser.add_argument("--pretrial", type=int, choices=[0, 1], default=None, help="Pretrial flag for clean full-population CLI mode.")
args = parser.parse_args()


how_many_divisions = args.how_many_divisions if args.how_many_divisions is not None else args.task_bins
pretrial_y_or_n = args.pretrial_y_or_n if args.pretrial_y_or_n is not None else args.pretrial
if how_many_divisions is None:
    raise ValueError("Provide the task-bin count either positionally or with --task_bins.")
if pretrial_y_or_n is None:
    raise ValueError("Provide the pretrial flag either positionally or with --pretrial.")

labels_an_path = args.labelsAn or args.CSUSAn
labels_a1_path = args.labelsA1 or args.CSUSA1
labels_b1_path = args.labelsB1 or args.CSUSB1
if labels_an_path is None:
    raise ValueError("Provide the A(n) label file either positionally or with --labelsAn.")
if labels_b1_path is None:
    raise ValueError("Provide the B(1) label file either positionally or with --labelsB1.")
if bool(args.traceAn_full) != bool(args.traceB1_full):
    raise ValueError("Provide both --traceAn_full and --traceB1_full for full-population mode.")

full_population_mode = bool(args.traceAn_full and args.traceB1_full)
if full_population_mode:
    traceAn_geometry = np.transpose(cebra.load_data(file=args.traceAn_full))
    traceB1_geometry = np.transpose(cebra.load_data(file=args.traceB1_full))
    traceA1An_A1 = np.empty((0, traceAn_geometry.shape[1]), dtype=traceAn_geometry.dtype)
else:
    missing_legacy = [
        name
        for name, value in {
            "traceA1An_An": args.traceA1An_An,
            "traceAnB1_An": args.traceAnB1_An,
            "traceA1An_A1": args.traceA1An_A1,
            "traceAnB1_B1": args.traceAnB1_B1,
            "CSUSA1": labels_a1_path,
        }.items()
        if value is None
    ]
    if missing_legacy:
        raise ValueError(
            "Matched-population legacy mode requires: "
            + ", ".join(missing_legacy)
            + ". For full-population mode, provide both --traceAn_full and --traceB1_full."
        )
    traceA1An_An = np.transpose(cebra.load_data(file=args.traceA1An_An))
    traceAnB1_An = np.transpose(cebra.load_data(file=args.traceAnB1_An))
    traceA1An_A1 = np.transpose(cebra.load_data(file=args.traceA1An_A1))
    traceAnB1_B1 = np.transpose(cebra.load_data(file=args.traceAnB1_B1))
    traceAn_geometry = traceAnB1_An
    traceB1_geometry = traceAnB1_B1

CSUSAn = cebra.load_data(file=labels_an_path)[0, :].flatten()
CSUSA1 = cebra.load_data(file=labels_a1_path)[0, :].flatten() if labels_a1_path else np.array([])
CSUSB1 = cebra.load_data(file=labels_b1_path)[0, :].flatten()
trial_ids_A1 = cebra.load_data(file=args.trial_ids_A1).flatten() if args.trial_ids_A1 else None
trial_ids_B1 = cebra.load_data(file=args.trial_ids_B1).flatten() if args.trial_ids_B1 else None

traceAn_geometry, CSUSAn = filter_pretrial(
    traceAn_geometry,
    CSUSAn,
    pretrial_y_or_n,
)
traceA1An_A1, CSUSA1, trial_ids_A1 = filter_pretrial_optional_ids(
    traceA1An_A1,
    CSUSA1,
    trial_ids_A1,
    pretrial_y_or_n,
)
traceB1_geometry, CSUSB1, trial_ids_B1 = filter_pretrial_optional_ids(
    traceB1_geometry,
    CSUSB1,
    trial_ids_B1,
    pretrial_y_or_n,
)

CSUSAn = bin_csus(CSUSAn, how_many_divisions)
CSUSA1 = bin_csus(CSUSA1, how_many_divisions) if len(CSUSA1) else CSUSA1
CSUSB1 = bin_csus(CSUSB1, how_many_divisions)

dimensions = how_many_divisions + pretrial_y_or_n
parameter_set = parameter_sets[args.parameter_set_name]

run_geometry_preservation(
    traceAn_geometry,
    traceAn_geometry,
    traceA1An_A1,
    traceB1_geometry,
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
    save_branch=args.save_branch,
    comparison_mode="An_vs_B1_separately_trained_full_population" if full_population_mode else "An_vs_B1_separately_trained_matched_population",
    resume_checkpoint=args.resume_checkpoint,
    CSUSA1_trial_ids=trial_ids_A1,
    CSUSB1_trial_ids=trial_ids_B1,
)
