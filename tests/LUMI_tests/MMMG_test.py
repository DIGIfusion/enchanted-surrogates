import pytest
import os, sys
sys.path.append(os.getcwd() + "/src")
import run

def mmmg_test():
    config_file = os.path.join(os.getcwd(),'tests/LUMI_tests/configs/MMMGaussian_config.yaml')
    args = run.load_configuration(config_file)
    run.main(args)
    
if __name__ == '__main__':
    mmmg_test()
        
    