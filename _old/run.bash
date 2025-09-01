## This is intended to be used when one is already on the commandline of an interactive node within an HPC

module purge 
module load python-data                          # changes based on CLUSTER 
# source .venv/bin/activate                      # changes based on CLUSTER 

current_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH:$current_dir/src   # does not change!

config_file=tglf_config.yaml                     # changes based on USE CASE

echo $config_file                                # does not change!
python3 src/run.py -cf=$current_dir/configs/$config_file