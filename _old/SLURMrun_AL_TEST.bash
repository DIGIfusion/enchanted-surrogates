#!/bin/bash
# Call with `sbatch SLURMrun.bash` and modify below with your relevant SLURM config
#SBATCH --job-name=ENCHANTEDSURROGATE
#SBATCH --account=project_2007159
#SBATCH --time=8:00:00
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --out=./al_run_cpugpu.out

module purge 
module load python-data                          # changes based on CLUSTER 
source .venv/bin/activate                      # changes based on CLUSTER 

current_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH:$current_dir/src   # does not change!
export PYTHONPATH=$PYTHONPATH:$current_dir/src/samplers/bmdal/

config_file=AL_example_gpucpu.yaml                     # changes based on USE CASE

echo $config_file                                # does not change!
python3 src/run.py -cf=$current_dir/configs/$config_file
# python3 src/run.py -cf=$current_dir/tests/configs/active_learning_STATICPOOL_ex_dset_SLURM.yaml