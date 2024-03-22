#!/bin/bash
#SBATCH --job-name=SURROGATE_WORKFLOW
#SBATCH --account=project_2005083
#SBATCH --time=01:00:00
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --out=./test.out

module purge 
module load python-data 

# source $WRKDIR/.venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$WRKDIR/src
srun python3 src/run.py