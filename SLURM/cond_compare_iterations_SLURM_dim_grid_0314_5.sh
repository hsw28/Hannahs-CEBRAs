#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3 ## number of jobs to run "in parallel"
#SBATCH --mem=30GB
#SBATCH --time=24:00:00
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



# Run the Python script with hardcoded arguments as sbatch ~/Programming/Hannahs-CEBRAs/SLURM/pos_compare_iterations_SLURM.sh
#python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/pos_compare_iterations_script.py ./traceA1An_An.mat ./traceAnB1_An.mat ./traceA1An_A1.mat ./traceAnB1_B1.mat ./PosAn.mat ./PosA1.mat ./PosB1.mat --iterations 10 --parameter_set_name set0222

python /home/hsw967/Programming/Hannahs-CEBRAs/scripts/cond_compare_iterations_script_dim_grid5.py ./traceA1An_An.mat ./traceAnB1_An.mat ./traceA1An_A1.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkA1.mat ./eyeblinkB1.mat 5 0 --iterations 25 --parameter_set_name set0314 --dimensions 2,3,5,7,10
