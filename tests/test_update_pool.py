import pytest
import os
import sys
import numpy as np

sys.path.append(os.getcwd() + "/src/samplers/bmdal")
sys.path.append(os.getcwd() + "/src")
import run
import samplers


config_filepath2 = os.path.join(
    os.getcwd(), "configs/AL_Streaming_QLK_gpucpu.yaml"
)

config_filepath = os.path.join(
    os.getcwd(), "tests/configs/active_learning_STATICPOOL_ex_dset.yaml"
)

configs_to_test = [config_filepath, config_filepath2]


@pytest.mark.parametrize("cfg", configs_to_test)
def test_run_active_learning(cfg):
    args = run.load_configuration(cfg)
    sampler = getattr(samplers, args.sampler.pop("type"))(**args.sampler)

    new_idxs = np.arange(10)
    sampler.parser.update_pool_and_train(new_idxs)
    
    assert True
