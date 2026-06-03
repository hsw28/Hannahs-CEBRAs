#!/bin/bash
#SBATCH --account=p32072
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20GB
#SBATCH --time=48:00:00
#SBATCH --job-name="geom_0222_${SLURM_ARRAY_TASK_ID}"
#SBATCH --output=geometry_preservation_0222.%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hsw@northwestern.edu

module purge
eval "$(conda shell.bash hook)"
source activate ratinabox

export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/Hannahs-CEBRAs"
export PYTHONPATH="${PYTHONPATH}:/home/hsw967/Programming/Hannahs-CEBRAs/scripts"

#python /home/hsw967/Programming/Hannahs-CEBRAs/scripts/cond_geometry_preservation_script.py --traceAn_full ./traceAn.mat --traceB1_full ./traceB1.mat --labelsAn ./eyeblinkAn.mat --labelsB1 ./eyeblinkB1.mat --task_bins 5 --pretrial 0 --iterations 20 --shuffles 1 --parameter_set_name set0222 --rat_id rat0222

python /home/hsw967/Programming/Hannahs-CEBRAs/scripts/cond_geometry_preservation_script.py --traceAn_full ./traceAn.mat --traceB1_full ./traceB1.mat --labelsAn ./eyeblinkAn.mat --labelsB1 ./eyeblinkB1.mat --task_bins 5 --pretrial 0 --iterations 20 --shuffles 0 --parameter_set_name set0222 --rat_id rat0222
