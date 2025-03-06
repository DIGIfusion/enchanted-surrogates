import pytest
import os
import sys
sys.path.append(os.getcwd() + "/src")
import run

def test_slurm_executor():
    config_filepath = os.path.join(test_config_dir, config_name)
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    if "ActiveLearning" in args.sampler["type"]:
        args.sampler["parser_kwargs"]["data_path"] = data_path
        args.executor['base_run_dir'] += '_simple'
    run.main(args)
    assert True
