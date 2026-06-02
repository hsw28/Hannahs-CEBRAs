#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --time=48:00:00
#SBATCH --job-name="geom_0313_${SLURM_ARRAY_TASK_ID}"
#SBATCH --output=geometry_preservation_0313.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hsw@northwestern.edu

module purge
eval "$(conda shell.bash hook)"
source activate ratinabox

export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/Hannahs-CEBRAs"
export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/Hannahs-CEBRAs/scripts"

#python /home/hsw967/Programming/Hannahs-CEBRAs/scripts/cond_geometry_preservation_script.py ./traceA1An_An.mat ./traceAnB1_An.mat ./traceA1An_A1.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkA1.mat ./eyeblinkB1.mat 5 0 --iterations 20 --shuffles 1 --parameter_set_name set0313 --rat_id rat0313

python /home/hsw967/Programming/Hannahs-CEBRAs/scripts/cond_geometry_preservation_script.py ./traceA1An_An.mat ./traceAnB1_An.mat ./traceA1An_A1.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkA1.mat ./eyeblinkB1.mat 5 0 --iterations 20 --shuffles 0 --parameter_set_name set0313 --rat_id rat0313 --traceAn_full ./traceAn.mat --traceB1_full ./traceB1.mat
