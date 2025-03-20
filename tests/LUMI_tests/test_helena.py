import pytest
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# sys.path.append(os.getcwd() + "/src/samplers/bmdal")
sys.path.append(os.getcwd() + "/src")
from parsers.HELENAparser import HELENAparser
import run

from runners.HELENArunner import HELENArunner


#Placing this before a function will trigger pytest to run the test multiple times and inserting a different value from the list into the variable each time
#list_of_values = [2,2,2,2]
#@pytest.mark.parametrize("variable", list_of_values)
#def test_example(variable):
    # assert variable = 2

def test_helena_basic():
    parser = HELENAparser()
    config_filepath = "tests/LUMI_tests/helena_config.yaml"
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    run.main(args)
    base_run_dir = args.executor["base_run_dir"]
    
    run_directories = [name for name in os.listdir(base_run_dir) if os.path.isdir(os.path.join(base_run_dir, name))]
    run_directories = [os.path.join(base_run_dir, run_dir) for run_dir in run_directories]
    test = []
    files_to_check = ['fort.10', 'fort.20', 'fort.30', 'fort.12']
    for run_dir in run_directories:
        files = os.listdir(run_dir)
        for check_file in files_to_check:
            if not check_file in files:
                raise FileNotFoundError(f"This file was not made by HELENA, {check_file}, suggesting HELENA did run run correctly.")
    
    success_s = []
    for run_dir in run_directories:
        success, mercier_stable, ballooning_stable = parser.read_output_file(run_dir)
        success_s.append(success)
    
    if all(success_s):
        assert True
    else:
        print('test failed because ALPHA1 and MERCIER could not be found in the fort.20 file by the HELENAparser.read_output_file() in all of the run directories')
        assert False

def test_helena_noKBM_betaN():
    # with dask
    config_filepath = "tests/LUMI_tests/helena_noKBM_betaN_config.yaml"
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    base_run_dir = args.executor["base_run_dir"]
    print('DELETEING WHAT IS IN BASE RUN DIR BEFORE RUNNING:\n',base_run_dir)
    os.system(f'rm -rf {base_run_dir}/*')
    
    print('STARTING ENCHANTED SURROGATES MAIN')
    run.main(args)
    print('FINISHED ENCHANTED SURROGATES MAIN')
    
    run_directories = [name for name in os.listdir(base_run_dir) if os.path.isdir(os.path.join(base_run_dir, name))]
    run_directories = [os.path.join(base_run_dir, run_dir) for run_dir in run_directories]
    
    beta_tolerance = args.runner['other_params']['beta_tolerance']
    beta_target = args.runner['other_params']['beta_N_target']
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
    
    failure = np.abs(np.array(beta_N_s)-beta_target) > beta_tolerance * beta_N_s
    if any(failure): #The itterations weren't enough to get within tolerance
        raise Exception(f'There have been {np.sum(failure)} failures in reaching the set tolerance within {max_beta_iterations} beta iterations')

        # files = os.listdir(run_dir)
        # for check_file in files_to_check:
        #     if not check_file in files:
        #         raise FileNotFoundError(f"This file was not made by HELENA, {check_file}, suggesting HELENA did run run correctly.")
    else:
        assert True
        return True

def test_helena_runner():
    
    # Current Issue:
    # There is a problem with the beta iteration, It kept going lower away from the target. You should print the variables involved in setting the new value based on the old value. 
    
    # without dask
    # runner:
    #   type: HELENArunner
    #   executable_path: "/project/project_462000451/HELENA/bin/hel13_64"
    #   other_params: {
    #     # "namelist_path": "/project/project_462000451/jet_97781_data/97781_T029_fort.10",
    #     "namelist_path": "/projappl/project_462000451/aarojarvinen/hel_test_runt/fort.10",
    #     "only_generate_files": False,
    #     #"beta_iteration": True,
    #     "input_parameter_type": 7,
    #     "beta_iteration": 1,
    #     "beta_tolerance": 0.5,
    #     "max_beta_iterations": 5,
    #     "constant_beta": 2.555}
    #     #Computed using core temp and density with magnetic field strength computed by GENE.
    executable_path = "/project/project_462000451/HELENA/bin/hel13_64"
    other_params = {
        "namelist_path": "/projappl/project_462000451/aarojarvinen/hel_test_runt/fort.10",
        "only_generate_files": False,
        "input_parameter_type": 7,
        "beta_iteration": 1,
        "input_value_1": 0.0,
        "input_value_2": 10.0,
        "beta_iterations_afp": False,
        "beta_tolerance": 0.01, # changed to absolute tolerence
        "max_beta_iterations": 5,
        "beta_N_target": 2.555 # 
        }
    runner = HELENArunner(executable_path, other_params)
    # bounds: [[1.0, 1.8], [3.0, 4.4], [0.05, 0.1], [0.5, 2.8]]
    #   num_samples: [2, 1, 1, 1]
    #   parameters: ['T_eped', 'n_eped', 'd_n_ped', 'n_esep']
    
    sample = {'T_eped':1.5, 'n_eped':3.5, 'd_n_ped':0.07, 'n_esep':1}
    run_dir = "/scratch/project_462000451/daniel/sprint_out/helena_beta/notebook_test"
    os.system(f'rm -rf {run_dir}/*')
    runner.single_code_run(sample, run_dir)
    parser = HELENAparser()
    success1, mercier_stable, ballooning_stable = parser.read_output_file(run_dir)
    
    beta_tolerance = other_params['beta_tolerance']
    beta_target = other_params['beta_N_target']
    fort20 = os.path.join(run_dir, 'fort.20')
    with open(fort20, 'r') as file:
        for line in file:
            if 'NORM. BETA' in line:
                break
        beta_N = float(line.strip().split(' ')[-1])*1e2
    success2 = np.abs(beta_N-beta_target) < beta_tolerance
    success = success1 and success2
    assert success
    
'''
BETA ITERATION FINISHED.
Target betaN: 2.555
Final betaN: 2.544
'''

if __name__ == '__main__':
    # test_example_configs()
    # print('TEST RESULT:',test_helena_noKBM_betaN())
    test_helena_runner()