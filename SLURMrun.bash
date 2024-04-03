#!/bin/bash
# Call with `sbatch SLURMrun.bash` and modify below with your relevant SLURM config
#SBATCH --job-name=SURROGATE_WORKFLOW
#SBATCH --account=project_2007159
#SBATCH --time=01:00:00
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --out=./run.out

module purge 
module load python-data                          # changes based on CLUSTER 
# source .venv/bin/activate                      # changes based on CLUSTER 

current_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH:$current_dir/src   # does not change!

config_file=tglf_config.yaml                     # changes based on USE CASE

echo $config_file                                # does not change!
python3 src/run.py -cf=$current_dir/configs/$config_file