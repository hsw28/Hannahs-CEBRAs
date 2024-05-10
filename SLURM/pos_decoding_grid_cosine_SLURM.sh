#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-4 ## number of jobs to run "in parallel"
#SBATCH --mem=220GB
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


# Run the Python script with hardcoded arguments as sbatch ~/Programming/Hannahs-CEBRAs/SLURM/pos_compare_iterations_SLURM.sh

python /home/hsw967/Programming/Hannahs-CEBRAs/scripts/pos_decoding_AvsB_grid.py ./traceA1An_An.mat ./traceA1An_A1.mat ./posAn.mat ./posA1.mat --learning_rate .00775,.001,.0008875,.000775,0.0006625,.00055 --min_temperature .000000001,.05,0.1,.15,.2,.25 --max_iterations 8000,10000,12000,14000,16000,18000
