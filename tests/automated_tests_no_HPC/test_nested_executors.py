import pytest
import os
import sys

sys.path.append(os.getcwd() + "/src/samplers/bmdal")
sys.path.append(os.getcwd() + "/src")
import run


config_filepath1 = os.path.join(
    os.getcwd(), "tests/automated_tests_no_HPC/configs/Nested_SimulationExecutor_MMMGaussian.yaml"
)

data_path = os.path.join(os.getcwd(), "tests", "automated_tests_no_HPC", "train.csv")


configs_to_test = [config_filepath1]


@pytest.mark.parametrize("config_name", configs_to_test)
def test_run_active_learning(config_name):
    args = run.load_configuration(config_name)
    args.executor["config_filepath"] = config_name
    os.system(f"rm -r {args.executor['base_run_dir']}")
    run.main(args)
    assert True
