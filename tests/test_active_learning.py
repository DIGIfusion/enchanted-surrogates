import pytest
import os
import sys
sys.path.append(os.getcwd() + "/src")
import samplers
import run


config_filepath2 = os.path.join(os.getcwd(), "tests/configs/active_learning_STATICPOOL.yaml")

data_path = os.path.join(os.getcwd(), "tests/train.csv")


configs_to_test = [config_filepath2]

@pytest.mark.parametrize("config_name", configs_to_test)
def test_run_active_learning(config_name):
    args = run.load_configuration(config_name)
    args.executor["config_filepath"] = config_name
    if args.sampler['type'] == 'ActiveLearningStaticPoolSampler':
        args.sampler['parser_kwargs']['data_path'] = data_path
    run.main(args)
    assert True

