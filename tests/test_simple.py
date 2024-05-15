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

test_config_dir = os.path.join(os.getcwd(), "tests/configs")
configs_to_test = os.listdir(test_config_dir)
@pytest.mark.parametrize("config_name", configs_to_test)
def test_run_simple_sampler(config_name): 
    config_filepath = os.path.join(test_config_dir, config_name)
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    run.main(args)
    assert True

