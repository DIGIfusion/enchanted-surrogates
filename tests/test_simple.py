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

    
def test_simple_grid_localexecutor():
    config_filepath = os.path.join(os.getcwd(), "tests/configs/grid.yaml")
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    run.main(args)
    assert True

def test_simple_random_seq_localexecutor():
    config_filepath = os.path.join(os.getcwd(), "tests/configs/random_seq.yaml")
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath