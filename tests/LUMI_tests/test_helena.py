import pytest
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# sys.path.append(os.getcwd() + "/src/samplers/bmdal")
sys.path.append(os.getcwd() + "/src")
from parsers.HELENAparser import HELENAparser
import run

#Placing this before a function will trigger pytest to run the test multiple times and inserting a different value from the list into the variable each time
#list_of_values = [2,2,2,2]
#@pytest.mark.parametrize("variable", list_of_values)
#def test_example(variable):
    # assert variable = 2

# def test_helena_basic():
#     parser = HELENAparser()
#     config_filepath = "tests/LUMI_tests/helena_config.yaml"
#     args = run.load_configuration(config_filepath)
#     args.executor["config_filepath"] = config_filepath
#     run.main(args)
#     base_run_dir = args.executor["base_run_dir"]
    
#     run_directories = [name for name in os.listdir(base_run_dir) if os.path.isdir(os.path.join(base_run_dir, name))]
#     run_directories = [os.path.join(base_run_dir, run_dir) for run_dir in run_directories]
#     test = []
#     files_to_check = ['fort.10', 'fort.20', 'fort.30', 'fort.12']
#     for run_dir in run_directories:
#         files = os.listdir(run_dir)
#         for check_file in files_to_check:
#             if not check_file in files:
#                 raise FileNotFoundError(f"This file was not made by HELENA, {check_file}, suggesting HELENA did run run correctly.")
    
#     success_s = []
#     for run_dir in run_directories:
#         success, mercier_stable, ballooning_stable = parser.read_output_file(run_dir)
#         success_s.append(success)
    
#     if all(success_s):
#         assert True
#     else:
#         print('test failed because ALPHA1 and MERCIER could not be found in the fort.20 file by the HELENAparser.read_output_file() in all of the run directories')
#         assert False

def test_helena_noKBM_betaN():
    config_filepath = "tests/LUMI_tests/helena_noKBM_betaN_config.yaml"
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    print('STARTING ENCHANTED SURROGATES MAIN')
    run.main(args)
    print('FINISHED ENCHANTED SURROGATES MAIN')
    base_run_dir = args.executor["base_run_dir"]
    
    run_directories = [name for name in os.listdir(base_run_dir) if os.path.isdir(os.path.join(base_run_dir, name))]
    run_directories = [os.path.join(base_run_dir, run_dir) for run_dir in run_directories]
    
    beta_tolerance = args.runner['other_params']['beta_tolerance']
    beta_target = args.runner['other_params']['constant_beta']
    max_beta_iterations = args.runner['other_params']['max_beta_iterations']
    beta_N_s = []
    for run_dir in run_directories:
        fort20 = os.path.join(run_dir, 'fort.20')
        with open(fort20, 'r') as file:
            for line in file:
                if 'NORM. BETA' in line:
                    break
            beta_N = line.strip().split(' ')[-1]
            beta_N_s.append(float(beta_N))
    
    failure = np.abs(np.array(beta_N_s)-beta_target) > beta_tolerance
    if any(failure): #The itterations weren't enough to get within tolerance
        raise Exception(f'There have been {np.sum(failure)} failures in reaching the set tolerance within {max_beta_iterations} beta iterations')

        # files = os.listdir(run_dir)
        # for check_file in files_to_check:
        #     if not check_file in files:
        #         raise FileNotFoundError(f"This file was not made by HELENA, {check_file}, suggesting HELENA did run run correctly.")
    else:
        assert True
        return True

if __name__ == '__main__':
    # test_example_configs()
    print('TEST RESULT:',test_helena_noKBM_betaN())