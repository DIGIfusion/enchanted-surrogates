import pytest
import os
import sys

sys.path.append(os.getcwd() + "/src")
import run
configs_to_test = ['tests/LUMI_tests/configs/gene_scan_runner.yaml']
@pytest.mark.parametrize("config_path", configs_to_test)
def test_example_configs(config_path):
    config_full_path = os.path.join(os.getcwd(), config_path)
    args = run.load_configuration(config_full_path)
    run.main(args)
    
if __name__ == '__main__':
    test_example_configs('tests/LUMI_tests/configs/gene_scan_runner.yaml')
    
    