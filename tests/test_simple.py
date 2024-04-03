import pytest
import os
import sys
import argparse

sys.path.append(os.getcwd() + "/src")
from runners import SIMPLErunner
from executors import LocalDaskExecutor
import run


def test_simple_initialization():
    runner = SIMPLErunner(executable_path="tests/simple/simple.sh")
    assert runner.single_code_run(params=[1, 2, 3], run_dir=".")


def test_simple_localexecutor():
    """ """
    config = {
        "sampler": {
            "type": "HypercubeSampler",
            "bounds": [[1, 3], [1, 3]],
            "num_samples": 3,
            "parameters": ["m", "n"],
        },
        "runner": {
            "type": "SIMPLErunner",
            "executable_path": "tests/simple/simple.sh",
            "other_params": {},
        },
        "executor": {
            "type": "LocalDaskExecutor",
            "base_run_dir": "simple_test_runs/",
            "worker_args": {
                "account": "project_2009007",
                "queue": "medium",
                "cores": 2,
                "memory": "1GB",
                "processes": 1,
                "walltime": "00:10:00",
                "interface": "ib0",
                "job_script_prologue": [
                    "export PYTHONPATH=$PYTHONPATH:/src",
                ],
            },
            "num_workers": 2,
        },
    }
    args = argparse.Namespace(**config)
    run.main(args)
    assert True


def test_missing_config():
    """ """
    config_file = ""
    with pytest.raises(FileNotFoundError):
        args = run.load_configuration(config_file)
        run.main(args)
