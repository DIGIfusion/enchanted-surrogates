import os, sys
import json


this_file_name, config_file, profile_file = sys.argv

with open(profile_file, 'r') as file:
    profile = json.load(file)

for env_var in profile['env'].keys():
    os.environ[env_var] = profile['env'][env_var]

# Still needs to be ran before running this python file
#       This python file should be converted to a bash script
for env_var in profile['env'].keys():
    os.environ[env_var] = profile['env'][env_var]
    
# os.environ['PATH'] = os.environ['PATH'] + profile['env']['ENCHANTED_ENV_PATH']
#------------------------------------------------------
print('debug',profile['env']['ENCHANTED_SRC_PATH'])
sys.path.append(profile['env']['ENCHANTED_SRC_PATH']) 

import run
args = run.load_configuration(config_file)
run.main(args)