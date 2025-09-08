# How to test on LUMI
The tests require you to have some setup on the lumi machine. 
Currently you need to have some way to activate a python enviroment.
The csc approved way is to use containerised python enviroments, see: https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/

You need to have a cloned version of the enchanted surrogates repo, see: https://github.com/DIGIfusion/enchanted-surrogates

You then need to make a copy of the user_config_<your-lumi-username>.json found in tests/LUMI_tests/user_config_<your-lumi-username>.json

Rename your copied file to have your lumi username in the indicated position.

Place *<your-lumi-username>* into your cd .git/info/exclude file so any personal files named with your username cannot be committed and pushed online. (It may be difficult to find on vs studio, try using a terminal based editor)

You can then run the LUMI_tests by:  
 - activating your python enviroment  
´export PATH=/scratch/project_XXXXXXXXX/your_container_folder/bin:$PATH´

 - starting an interactive session  
´srun --account=project_XXXXXXXXX --partition=small --ntasks=2 --time=04:00:00 --mem=2GB --pty bash´

 - Running pytest from the cloned enchanted surrogates repo  
´python3 -m pytest tests/LUMI_tests´
  