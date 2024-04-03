#!/bin/bash
#SBATCH --job-name=SURROGATE_WORKFLOW
#SBATCH --account=project_2007159
#SBATCH --time=01:00:00
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --out=./test.out

module purge 
module load python-data 

# source $WRKDIR/.venv/bin/activate

current_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH:$current_dir/src

config_file=$current_dir/configs/tglf_config.yaml

echo $config_file
srun python3 src/run.py -cf=$config_file