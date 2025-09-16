import os
import glob
import shutil

from ..utils.append_es_to_path import append_es_to_path
append_es_to_path()
from enchanted_surrogates.executors import LocalExecutor, JoblibExecutor


# https://docs.pytest.org/en/stable/how-to/tmp_path.html
def test_full_workflow_local(tmp_path):
    config = {}
    # -- sampler
    # TODO: test different samplers
    bounds = [[-5, 5], [0, 1]]
    parameters = ['c1', 'c2']
    budget = 10
    sampler_kwargs = {
        'type': 'RandomSampler', 'bounds': bounds, 'budget': budget, 'parameters': parameters}
    runner_kwargs = {
        'type': 'ExampleRunner'
    }

    base_run_dir = tmp_path  # f"{os.path.dirname(__file__)}/example"
    # create the executor

    executor = LocalExecutor(
        sampler_kwargs=sampler_kwargs,
        runner_kwargs=runner_kwargs,
        base_run_dir=base_run_dir,
        **config)
    executor.start_runs()
    # This should create {budget} folders with ??? inside

    assert len(glob.glob(os.path.join(base_run_dir, "*"))) == budget
    executor.clean()

    # Clean up test
    shutil.rmtree(base_run_dir)


def test_full_workflow_joblib(tmp_path):
    config = {}
    # -- sampler
    # TODO: test different samplers
    bounds = [[-5, 5], [0, 1]]
    parameters = ['c1', 'c2']
    budget = 50
    sampler_kwargs = {
        'type': 'RandomSampler', 'bounds': bounds, 'budget': budget, 'parameters': parameters}

    # -- runner args
    runner_kwargs = {
        'type': 'ExampleRunner'
    }

    base_run_dir = tmp_path  # f"{os.path.dirname(__file__)}/example"
    # create the executor
    executor = JoblibExecutor(
        sampler_kwargs=sampler_kwargs,
        runner_kwargs=runner_kwargs,
        base_run_dir=base_run_dir,
        **config)
    executor.start_runs() 
    # This should create {budget} folders with ??? inside

    created_rundirs = glob.glob(os.path.join(base_run_dir, "*"))
    assert len(created_rundirs) == budget
    executor.clean()
