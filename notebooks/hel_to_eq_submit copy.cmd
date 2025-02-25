#!/bin/bash -l
## LUMI-C (CPU partition) submit script template
## Submit via: sbatch submit.cmd (parameters below can be overwritten by command line options)
#SBATCH -t 01:00:00                # wallclock limit
#SBATCH -N 1                       # total number of nodes, 2 CPUs with 64 rank each
#SBATCH --ntasks=1      # 64 per CPU (i.e. 128 per node). Additional 2 hyperthreads disabled
#SBATCH --mem=500MB                    # Allocate all the memory on each node
#SBATCH -p small                # all options see: scontrol show partition
#SBATCH -J hel_to_eqdsk                    # Job name
#SBATCH -o ./slurm_out/%x.%j.out
#SBATCH -e ./slurm_out/%x.%j.err
##uncomment to set specific account to charge or define SBATCH_ACCOUNT/SALLOC_ACCOUNT globally in ~/.bashrc
#SBATCH -A project_462000451

## Activate Python Enviroment
source /scratch/project_462000451/daniel/daniel_sprint/bin/activate

## Run the python script 
## $1 needs to be fpath, $2 needs to be eqdsk_path
## they are specified like this: sbatch hel_to_eq_submit.cmd path/to/fpath path/to/eqdsk_path

echo 'Showing bash variables for the fpath and eqdisk_path'
echo $1
echo $2

python3 /project/project_462000451/enchanted-surrogates_11feb2025/notebooks/helena_to_eqdsk.py $1 $2

echo 'COMPLETE'