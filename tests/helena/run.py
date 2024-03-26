# run.py
import yaml 
import sys
sys.path.append('/scratch/project_2009007/enchanted-surrogates/src')
import samplers
import runners 
import executors 

def load_configuration(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(args):
    sampler = getattr(samplers, args.sampler.pop('type'))(**args.sampler) 
    # runner = getattr(runners, args.runner.pop('type'))(**args.runner) 
    executor = getattr(executors, args.executor.pop('type'))(sampler, args.runner, **args.executor)  

    # executor.initialize(sampler=sampler, runner=runner, parser=parser)
    executor.start_runs()

import argparse 
if __name__ == "__main__":
    # TODO: this should be argument with argparse
    config_path = "/scratch/project_2009007/enchanted-surrogates/tests/helena/test.yaml"
    config = load_configuration(config_path) 
    args = argparse.Namespace(**config) #Daniel: I don't understant why you need to convert it to a namespace. Isn't main(**config), just fine??
    main(args)
