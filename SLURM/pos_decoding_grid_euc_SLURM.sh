#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=36:00:00
#SBATCH --job-name=cebra_grid
#SBATCH --output=AM_SLURM_out.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hsw@northwestern.edu

module purge
eval "$(conda shell.bash hook)"
source activate ratinabox
#i have cebra installed in ratinabox

export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/Hannahs-CEBRAs"
export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/Hannahs-CEBRAs/scripts"


# Run the Python script with hardcoded arguments as sbatch ~/Programming/Hannahs-CEBRAs/SLURM/pos_compare_iterations_SLURM.sh

srun python -u "${PROJECT}/scripts/pos_decoding_AvsB_grid2.py" \
    ./traceA1An_An.mat \
    ./traceA1An_A1.mat \
    ./posAn.mat \
    ./posA1.mat \
    --learning_rate 0.00001,0.00005,0.0001 \
    --min_temperature 0.000000001,1.5 \
    --max_iterations 40000,80000,140000
