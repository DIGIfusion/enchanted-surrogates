import pytest
import os
import sys
sys.path.append(os.getcwd() + "/src")
import samplers
import run


config_filepath1 = os.path.join(os.getcwd(), "tests/configs/active_learning.yaml")
config_filepath2 = os.path.join(os.getcwd(), "tests/configs/active_learning_database.yaml")

config_filepaths = [config_filepath1, config_filepath2]

data_path = os.path.join(os.getcwd(), "tests/train.csv")
def test_horrible_from_file():     
    args = run.load_configuration(config_filepath1)
    assert args.sampler['kernel_transform'] == [['rp', [512]]]
    
def test_load_sampler_from_file():
    for config_filepath in config_filepaths:
        
        args = run.load_configuration(config_filepath)
        if args.sampler['type'] == 'ActiveLearningStaticPoolSampler':
            args.sampler['data_path'] = data_path
        sampler = getattr(samplers, args.sampler.pop('type'))(**args.sampler)
        print('\n', 100*'=')
        print(sampler.__class__)

        parameter_list = sampler.get_initial_parameters()
        assert len(parameter_list) == sampler.init_num_samples

def test_integrate_sampler_with_local_executor(): 
    for config_filepath in config_filepaths: 
        args = run.load_configuration(config_filepath)
        if args.sampler['type'] == 'ActiveLearningStaticPoolSampler':
            args.sampler['data_path'] = data_path
        args.executor["config_filepath"] = config_filepath
        run.main(args)

