import pytest
import os
import sys

sys.path.append(os.getcwd() + "/src")
import run
configs_to_test = ['tests/LUMI_tests/configs/simple_config_lumi_simulation.yaml','tests/LUMI_tests/configs/simple_config_lumi.yaml', 'tests/LUMI_tests/configs/simple_pipeline_config.yaml']
@pytest.mark.parametrize("config_path", configs_to_test)
def test_example_configs(config_path):
    config_full_path = os.path.join(os.getcwd(), config_path)
    args = run.load_configuration(config_full_path)
    run.main(args)
    
    