#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-2
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --job-name=pos_consistencyAB
#SBATCH --output=pos_consistencyAB.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hsw@northwestern.edu

# can run like sbatch /home/hsw967/Programming/Hannahs-CEBRAs/SLURM/pos_consistencyAB_SLURM.sh set0222

# Submit from ONE animal's data directory, e.g.:
# sbatch /home/hsw967/Programming/Hannahs-CEBRAs/SLURM/pos_consistencyAB_SLURM.sh set0222
# Valid animals: set0222, set0307, set0313, set0314, set0816.
# Array task 1 runs parameter group 1; task 2 runs group 2 (no duplicate runs).
# Each task: 20 fits each for An, B1, An_shuffled, B1_shuffled; default dimension 3.
# Saves all 80 run embeddings for later geometry analysis, plus the four best models.
# Additional Python options can follow the animal, e.g. --dimensions 2 3 5 7 10.
# A dimension sweep runs sequentially within each task and may need more walltime.
# To run one parameter group only, submit with sbatch --array=1 (or --array=2).
# Optional environment overrides: REPO_DIR, DATA_DIR, TRACE_AN, TRACE_B1,
# POS_AN, POS_B1, OUTPUT_DIR. Relative input overrides are relative to DATA_DIR.

set -eo pipefail

parameter_set_name="${1:-}"
case "$parameter_set_name" in
    set0222|set0307|set0313|set0314|set0816) ;;
    *) echo "Usage: sbatch $0 set0222|set0307|set0313|set0314|set0816 [Python options]" >&2; exit 2 ;;
esac
shift
parameter_group="${SLURM_ARRAY_TASK_ID:?Submit with sbatch as an array job}"
case "$parameter_group" in
    1|2) ;;
    *) echo "Array task must be 1 or 2 (parameter group)." >&2; exit 2 ;;
esac

REPO_DIR="${REPO_DIR:-/home/hsw967/Programming/Hannahs-CEBRAs}"
DATA_DIR="${DATA_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$DATA_DIR"
TRACE_AN="${TRACE_AN:-traceAnB1_An.mat}"
TRACE_B1="${TRACE_B1:-traceAnB1_B1.mat}"
# Existing cluster scripts use posAn/posB1; also accept PosAn/PosB1.
if [[ -z "${POS_AN:-}" ]]; then
    POS_AN=posAn.mat
    [[ -f "$POS_AN" ]] || POS_AN=PosAn.mat
fi
if [[ -z "${POS_B1:-}" ]]; then
    POS_B1=posB1.mat
    [[ -f "$POS_B1" ]] || POS_B1=PosB1.mat
fi
for input in "$TRACE_AN" "$TRACE_B1" "$POS_AN" "$POS_B1"; do
    [[ -f "$input" ]] || { echo "Missing input: $DATA_DIR/$input" >&2; exit 2; }
done
[[ -f "$REPO_DIR/scripts/pos_consistencyAB_script.py" ]] || {
    echo "Missing position consistency runner in $REPO_DIR" >&2; exit 2;
}
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/position_consistency_results/job_${SLURM_ARRAY_JOB_ID}_${parameter_set_name}}"

module purge
eval "$(conda shell.bash hook)"
conda activate ratinabox
export PYTHONPATH="$REPO_DIR:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

echo "Animal: $parameter_set_name; parameter group: $parameter_group; data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
python "$REPO_DIR/scripts/pos_consistencyAB_script.py" \
    "$TRACE_AN" "$TRACE_B1" "$POS_AN" "$POS_B1" \
    --iterations 20 \
    --dimensions 3 \
    --seed 42 \
    --device cuda \
    --output-dir "$OUTPUT_DIR" \
    "$@" \
    --parameter_set_name "$parameter_set_name" \
    --parameter-group "$parameter_group"
