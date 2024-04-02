import pytest
import os
import sys

sys.path.append(os.getcwd() + "/src")
from runners import SIMPLErunner
import run


def test_simple_initialization():
    runner = SIMPLErunner(executable_path="tests/simple/simple.sh")
    assert runner.single_code_run(params=[1, 2, 3], run_dir=".")


def test_simple_local():
    """ """
    config_file = "tests/simple/test.yaml"
    args = run.load_configuration(config_file)
    run.main(args)
    assert True


def test_missing_config():
    """ """
    config_file = ""
    with pytest.raises(FileNotFoundError):
        args = run.load_configuration(config_file)
        run.main(args)
