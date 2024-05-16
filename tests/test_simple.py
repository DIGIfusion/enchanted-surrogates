import pytest
import os
import sys
import argparse
import yaml

sys.path.append(os.getcwd() + "/src")
from runners import SIMPLErunner
from executors import LocalDaskExecutor
import run


# def test_simple_initialization():
#     runner = SIMPLErunner(executable_path="tests/simple/simple.sh")
#     assert runner.single_code_run(params={'a': 1, 'b': 2, 'c':3}, run_dir="./simple_test_runs")

data_path = os.path.join(os.getcwd(), "tests/train.csv")
test_config_dir = os.path.join(os.getcwd(), "tests/configs")
configs_to_test = os.listdir(test_config_dir)


@pytest.mark.parametrize("config_name", configs_to_test)
def test_example_configs(config_name):
    config_filepath = os.path.join(test_config_dir, config_name)
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    if args.sampler['type'] == 'ActiveLearningStaticPoolSampler':
        args.sampler['data_path'] = data_path
    run.main(args)
    assert True

