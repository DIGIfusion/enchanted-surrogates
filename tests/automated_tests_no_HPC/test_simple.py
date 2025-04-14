import pytest
import os
import sys

sys.path.append(os.getcwd() + "/src/samplers/bmdal")
sys.path.append(os.getcwd() + "/src")
import run
import re



def is_valid_uuid(uuid_string):
    # Define the regular expression pattern for a valid UUID
    pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    # Use the re.match function to check if the uuid_string matches the pattern
    match = re.match(pattern, uuid_string)
    # Return True if the string matches the pattern, otherwise False
    return bool(match)


# def test_simple_initialization():
#     runner = SIMPLErunner(executable_path="tests/simple/simple.sh")
#     assert runner.single_code_run(params={'a': 1, 'b': 2, 'c':3}, run_dir="./simple_test_runs")

data_path = os.path.join(os.getcwd(), "tests/automated_tests_no_HPC/train.csv")
test_config_dir = os.path.join(os.getcwd(), "tests/automated_tests_no_HPC/configs")
configs_to_test = os.listdir(test_config_dir)
configs_to_test = [
    conf
    for conf in configs_to_test
    if ("SLURM" not in conf and conf != "active_learning_STATICPOOL.yaml")
]


@pytest.mark.parametrize("config_name", configs_to_test)
def test_example_configs(config_name):
    config_filepath = os.path.join(test_config_dir, config_name)
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    if "ActiveLearning" in args.sampler["type"]:
        args.sampler["parser_kwargs"]["data_path"] = data_path
        args.executor['base_run_dir'] += '_simple'
    run.main(args)
    
def test_random_seq():
    config_path = os.path.join(os.getcwd(), "tests/automated_tests_no_HPC/configs/random_seq.yaml")
    args = run.load_configuration(config_path)
    base_run_dir = args.executor['base_run_dir']
    os.system(f'rm -r {base_run_dir}/*')
    sampler, executor = run.main(args)
    dirs = os.listdir(executor.base_run_dir)
    dirs = [d for d in dirs if is_valid_uuid(d)]
    assert len(dirs) == sampler.num_initial_points
