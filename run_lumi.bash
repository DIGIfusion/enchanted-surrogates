## This is intended to be used when one is already on the commandline of an interactive node within an HPC


module purge 

## On lumi to avoid many small python files slowing down the lustre filesystem you should have your python wrapped in a container 
## Follow these instructions: https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/
## Then paste the export command it provides here
export PATH="/project/project_462000451/enchanted_container_lumi3/bin:$PATH"

## Alternativly you could have an enviroment to activate (Not advised)
# source .venv/bin/activate                      # changes based on CLUSTER 

current_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH:$current_dir/src   # does not change!

config_file=simple_config_lumi.yaml                     # changes based on USE CASE

echo $config_file                                # does not change!
python3 src/run.py -cf=$current_dir/configs/$config_file