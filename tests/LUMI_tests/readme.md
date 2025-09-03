# How to test on LUMI
The tests require you to have some setup on the lumi machine. 
Currently you need to have some way to activate a python enviroment.
The csc approved way is to use containerised python enviroments, see: https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/

You need to have a cloned version of the enchanted surrogates repo, see: https://github.com/DIGIfusion/enchanted-surrogates

You then need to edit the user_config.json found in tests/LUMI_tests/user_config.json

An example is shown below:  
´´´
{
    "path_to_enchanted-surrogates": "/users/your_user/enchanted-surrogates/src/",
    "activate_env_command": "export PATH=/scratch/project_XXXXXXXXX/your_container_folder/bin:$PATH"
    "project":'project_XXXXXXXXX'
}
´´´
You can then run the LUMI_tests by:  
 - activating your python enviroment  
´export PATH=/scratch/project_XXXXXXXXX/your_container_folder/bin:$PATH´

 - starting an interactive session  
´srun --account=project_XXXXXXXXX --partition=small --ntasks=2 --time=04:00:00 --mem=2GB --pty bash´

 - Running pytest from the cloned enchanted surrogates repo  
´python3 -m pytest tests/LUMI_tests´

Please do not commit or push personal user_config.json files
  