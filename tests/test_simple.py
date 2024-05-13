import pytest
import os
import sys
import argparse
import yaml

sys.path.append(os.getcwd() + "/src")
from runners import SIMPLErunner
from executors import LocalDaskExecutor
import run


def test_simple_initialization():
    runner = SIMPLErunner(executable_path="tests/simple/simple.sh")
    assert runner.single_code_run(params={'a': 1, 'b': 2, 'c':3}, run_dir="simple_test_runs/")

# def test_simple_cube_localexecutor():
# 
#     config_filepath = os.path.join(os.getcwd(), "tests/test_config_cube.yaml")
#     args = run.load_configuration(config_filepath)
#     args.executor["config_filepath"] = config_filepath
#     run.main(args)
# 
#     assert True
# 

def test_simple_grid_localexecutor():

    config_filepath = os.path.join(os.getcwd(), "tests/test_config_grid.yaml")
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    run.main(args)

    assert True


def test_simple_random_seq_localexecutor():

    config_filepath = os.path.join(os.getcwd(), "tests/test_config_random_seq.yaml")
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    run.main(args)

    assert True

 
def test_simple_random_batch_localexecutor():

    config_filepath = os.path.join(os.getcwd(), "tests/test_config_random_batch.yaml")
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    run.main(args)

    assert True
     
def test_missing_config():
    """ """
    config_file = ""
    with pytest.raises(FileNotFoundError):
        args = run.load_configuration(config_file)
        run.main(args)
