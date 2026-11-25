"""Run either/both shared position parameter sets for one animal's An/B1 sessions.

Example (from the directory containing the MATLAB inputs):
    python /path/to/Hannahs-CEBRAs/scripts/pos_consistencyAB_script.py \
        traceAnB1_An.mat traceAnB1_B1.mat PosAn.mat PosB1.mat \
        --parameter_set_name set0222 --parameter-group both --iterations 20

Use --dimensions 2 3 5 7 10 for a dimension sweep (default: 3).
Raw input preprocessing follows pos_compare_iterations_script.py: transpose neural
traces, smooth x/y with sigma=2, take every second position sample, truncate excess
positions, and retain speed >=4 using a 0.1333-second interval. Speed is assigned
to the ending frame consistently. --prepared bypasses all these transformations.

An and B1 must already share a meaningful x/y coordinate system. The default
matching tolerance is 2 coordinate units; inspect position_matches.npz and adjust
--max-match-distance to your spatial resolution. Mutual nearest-position matching
keeps unique individual sample pairs; it does not average latent state centroids.
At least 100 pairs are required by default. This is an explicit adaptation from
the conditioning pipeline's rowwise comparison, not CEBRA's built-in 1D alignment.

Each group/dimension gets its own timestamped output directory: four selected
models, full embeddings with positions, training permutations, all final losses,
selected-run metadata, matching indices/distances, configuration manifest,
directional R2 CSVs, and a four-model heatmap (An/B1 and shuffled controls).
Scores are fitted/evaluated on the same matched rows, as in built-in CEBRA;
these are descriptive consistency scores, not held-out prediction accuracy.
The shuffled models are trained on permuted neural rows with fixed position labels
and evaluated on original neural rows, matching the conditioning control.
The heatmap colors cover 0-100%; exact values remain in CSV and annotations.
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pos_consistencyAB import parameter_sets, prepare_session, run_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['traceAn', 'traceB1', 'PosAn', 'PosB1']:
        parser.add_argument(name)
    parser.add_argument('--parameter_set_name', choices=list(parameter_sets()['1']), required=True)
    parser.add_argument('--parameter-group', choices=['1', '2', 'both'], default='both')
    parser.add_argument('--iterations', type=int, default=20, help='Independent fits per session and control.')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[3])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda_if_available')
    parser.add_argument('--prepared', action='store_true', help='Inputs already samples x neurons / samples x 2; skip smoothing, downsampling and speed filter.')
    parser.add_argument('--min-speed', type=float, default=4.)
    parser.add_argument('--max-match-distance', type=float, default=2., help='In position coordinate units (cm if inputs are cm).')
    parser.add_argument('--min-pairs', type=int, default=100)
    parser.add_argument('--max-iterations', type=int, help='Override training length for smoke tests; recorded in manifest.')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/position_consistency'))
    args = parser.parse_args()
    if args.iterations < 1 or min(args.dimensions) < 1 or args.min_pairs < 2 or args.max_match_distance <= 0 or args.seed < 0:
        parser.error('Runs, dimensions, matching distance and seed must be valid positive values (seed may be zero).')
    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error('--max-iterations must be positive.')
    import cebra
    sessions = [prepare_session(cebra.load_data(file=trace), cebra.load_data(file=position),
                prepared=args.prepared, min_speed=args.min_speed)
                for trace, position in [(args.traceAn, args.PosAn), (args.traceB1, args.PosB1)]]
    groups = ['1', '2'] if args.parameter_group == 'both' else [args.parameter_group]
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    for group in groups:
        for dimension in args.dimensions:
            config = dict(parameter_sets()[group][args.parameter_set_name], model_architecture='offset10-model',
                          batch_size=512, output_dimension=dimension, conditional='time_delta',
                          device=args.device, num_hidden_units=32, time_offsets=1, verbose=False)
            if args.max_iterations is not None:
                config['max_iterations'] = args.max_iterations
            destination = args.output_dir / ('%s_group%s_dim%d_%s' % (args.parameter_set_name, group, dimension, stamp))
            run_analysis(sessions, config, destination, args.iterations, args.seed, args.max_match_distance,
                         args.min_pairs, metadata={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
            print('Saved:', destination, flush=True)


if __name__ == '__main__':
    main()
