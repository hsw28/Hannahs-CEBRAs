#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0 ## number of jobs to run "in parallel"
#SBATCH --mem=100GB
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

python /home/hsw967/Programming/Hannahs-CEBRAs/scripts/pos_decoding_AvsB_grid2.py ./traceA1An_An.mat ./traceA1An_A1.mat ./posAn.mat ./posA1.mat --learning_rate .001,.005,.0085,0.000325,0.00055,.0009,.00003,.00008,.000002 --min_temperature .1,.25,.5,.75,1,1.33,1.66,2,2.33,2.66,3.0 --max_iterations 8000,12000,16000,20000,24000,28000,32000
