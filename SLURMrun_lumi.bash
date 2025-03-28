#!/bin/bash
# Call with `sbatch SLURMrun.bash` and modify below with your relevant SLURM config
#SBATCH --job-name=SURROGATE_WORKFLOW
#SBATCH --account=project_462000451
#SBATCH --time=12:00:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --out=./run.out

echo PURGING MODULES
module purge 

echo LOADING PYTHON
export PATH="/project/project_462000451/enchanted_container_lumi3/bin:$PATH"                          # changes based on CLUSTER 
# source .venv/bin/activate                      # changes based on CLUSTER 

echo ADDING SRC TO PYTHON PATH
current_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH:$current_dir/src   # does not change!

config_file=gene_config_lumi.yaml                     # changes based on USE CASE

echo RUNNING ENCHANTED SURROGATES WITH CONFIG FILE: $current_dir/configs/$config_file  # does not change!
python3 src/run.py -cf=$current_dir/configs/$config_file