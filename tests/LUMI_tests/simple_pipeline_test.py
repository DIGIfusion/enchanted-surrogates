import pytest
import os
import sys

sys.path.append(os.getcwd() + "/src")
import run


def test_example_configs():
    config_filepath = os.path.join(os.getcwd(), "/users/danieljordan/enchanted-surrogates2/configs/simple_pipeline_config.yaml")
    args = run.load_configuration(config_filepath)
    run.main(args)
    assert True
    
if __name__ == "__main__":
    test_example_configs()
    