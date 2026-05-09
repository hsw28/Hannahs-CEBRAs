#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0 ## number of jobs to run "in parallel"
#SBATCH --mem=22GB
#SBATCH --time=48:00:00
#SBATCH --job-name="sample_job_\${SLURM_ARRAY_TASK_ID}" ## use the task id in the name of the job
#SBATCH --output=AM_SLURM_out.%A_%a.out ## use the jobid (A) and the specific job index (a) to name your log file
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hsw@northwestern.edu  ## your email


module purge
eval "$(conda shell.bash hook)"
source activate ratinabox
#i have cebra installed in ratinabox

export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/Hannahs-CEBRAs"
export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/Hannahs-CEBRAs/scripts"

# THIS IS THE IMPORTANT PART
export PYTHONUNBUFFERED=1

# Run the Python script with hardcoded arguments as sbatch ~/Programming/Hannahs-CEBRAs/SLURM/pos_compare_iterations_SLURM.sh

python -u /home/hsw967/Programming/Hannahs-CEBRAs/scripts/cond_decoding_AvsB_grid_cos_auto.py ./traceAnB1_An.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkB1.mat 2 0 --learning_rate 0.00125,0.0015,0.00175,0.002 --min_temperature 2.75,3,3.25 --max_iterations 43000,50000,57000



#python /home/hsw967/Programming/Hannahs-CEBRAs/scripts/cond_decoding_AvsB_grid_cos_auto.py ./traceA1An_An.mat ./traceA1An_A1.mat ./eyeblinkAn.mat ./eyeblinkA1.mat 2 0 --learning_rate 0.0035,0.0075,0.0095,0.02,0.05 --min_temperature 1,2,2.33,2.66,3,3.5 --max_iterations 8000,9000,10000,13000,16000,25000
