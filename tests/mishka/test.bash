#!/bin/bash
#SBATCH --job-name=SURROGATE_WORKFLOW
#SBATCH --account=project_2009007
#SBATCH --time=00:10:00
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --out=./test.out
#SBATCH --err=./test.err

module purge 
module load python-data 

# source $WRKDIR/.venv/bin/activate
# export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$WRKDIR/src
srun python3 -u /scratch/project_2009007/enchanted-surrogates/tests/mishka/run.py