#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=36:00:00
#SBATCH --job-name=cebra_grid
#SBATCH --output=AM_SLURM_out.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hsw@northwestern.edu

set -eo pipefail

module purge
eval "$(conda shell.bash hook)"
conda activate ratinabox

set -u

PROJECT=/home/hsw967/Programming/Hannahs-CEBRAs

export PYTHONPATH="${PROJECT}:${PROJECT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Submit from the directory containing your four .mat files.
cd "$SLURM_SUBMIT_DIR"

echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $PWD"
nvidia-smi

srun python -u "${PROJECT}/scripts/pos_decoding_AvsB_grid.py" \
    ./traceA1An_An.mat \
    ./traceA1An_A1.mat \
    ./posAn.mat \
    ./posA1.mat \
    --learning_rate .0001,.00005 \
    --min_temperature 0.000000001,1.5 \
    --max_iterations 120000,160000,240000

  #  --learning_rate .0001,.00005,.00001,.000005,.000001 \
  #  --min_temperature 0.0000000010,.5,1,1.5 \
  #  --max_iterations 20000,40000,60000,80000

echo "Finished: $(date)"
