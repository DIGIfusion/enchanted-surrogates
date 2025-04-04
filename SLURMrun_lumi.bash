#!/bin/bash
# Call with `sbatch SLURMrun.bash` and modify below with your relevant SLURM config
#SBATCH --job-name=SURROGATE_WORKFLOW
#SBATCH --account=project_462000451
#SBATCH --time=04:00:00
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

config_file = $1
# config_file=tests/LUMI_tests/configs/Active_MMMGaussian_config.yaml # tests/LUMI_tests/configs/Active_GENE_config.yaml                     # changes based on USE CASE

echo RUNNING ENCHANTED SURROGATES WITH CONFIG FILE: $1  # does not change!
srun python3 -u src/run.py -cf=$1

        # libfabric.so.1 => /opt/cray/libfabric/1.15.2.0/lib64/libfabric.so.1 (0x00007f7543bb4000)