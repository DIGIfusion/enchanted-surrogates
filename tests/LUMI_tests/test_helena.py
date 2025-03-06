import pytest
import os
import sys

# sys.path.append(os.getcwd() + "/src/samplers/bmdal")
sys.path.append(os.getcwd() + "/src")
import run

#Placing this before a function will trigger pytest to run the test multiple times and inserting a different value from the list into the variable each time
#list_of_values = [2,2,2,2]
#@pytest.mark.parametrize("variable", list_of_values)
#def test_example(variable):
    # assert variable = 2

def test_example_configs():
    config_filepath = "tests/LUMI_tests/helena_config.yaml"
    args = run.load_configuration(config_filepath)
    args.executor["config_filepath"] = config_filepath
    run.main(args)
    assert True

if __name__ == '__main__':
    test_example_configs()