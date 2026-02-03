"""
Basic tests for the full workflow with different executors.
Each executor should be tested in a separate function to avoid conflicts."""

import os
import glob
import shutil

from types import SimpleNamespace
from enchanted_surrogates.executors import LocalExecutor, JoblibExecutor, DaskExecutor
from enchanted_surrogates.supervisor.supervisor import Supervisor


# https://docs.pytest.org/en/stable/how-to/tmp_path.html
def test_full_workflow_local(tmp_path):
    config = {}
    # -- sampler
    # TODO: test different samplers
    bounds = [[-5, 5], [0, 1]]
    parameters = ["c1", "c2"]
    budget = 10
    executor_config = {"type": "LocalExecutor"}
    sampler_config = {
        "type": "RandomSampler",
        "bounds": bounds,
        "budget": budget,
        "parameters": parameters,
    }
    runner_config = {"type": "ExampleRunner"}

    base_run_dir = tmp_path

    args = SimpleNamespace(
        executors={"e1": executor_config},
        samplers={"s1": sampler_config},
        runners={"r1": runner_config},
        supervisor={
            "base_run_dir": base_run_dir,
            "run_order": [{"executor": "e1", "sampler": "s1", "runner": "r1"}],
        },
    )

    supervisor = Supervisor(args)
    supervisor.start()

    # This should create {budget} folders with ??? inside
    assert len(next(os.walk(tmp_path))[1]) == budget


def test_full_workflow_joblib(tmp_path):
    config = {}
    # -- sampler
    # TODO: test different samplers
    bounds = [[-5, 5], [0, 1]]
    parameters = ["c1", "c2"]
    budget = 50
    executor_config = {"type": "JoblibExecutor"}
    sampler_config = {
        "type": "RandomSampler",
        "bounds": bounds,
        "budget": budget,
        "parameters": parameters,
    }
    runner_config = {"type": "ExampleRunner"}

    base_run_dir = tmp_path

    args = SimpleNamespace(
        executors={"e1": executor_config},
        samplers={"s1": sampler_config},
        runners={"r1": runner_config},
        supervisor={
            "base_run_dir": base_run_dir,
            "run_order": [{"executor": "e1", "sampler": "s1", "runner": "r1"}],
        },
    )

    supervisor = Supervisor(args)
    supervisor.start()
    # This should create {budget} folders with ??? inside

    assert len(next(os.walk(tmp_path))[1]) == budget


def test_seamless_sampling(tmp_path):
    config = {}
    # -- sampler
    # TODO: test different samplers
    bounds = [[-5, 5], [0, 1]]
    parameters = ["c1", "c2"]
    budget = 50
    kill_after = 4
    batch_size = 10
    executor_config = {"type": "JoblibExecutor"}
    sampler_config = {
        "type": "RandomSamplerWithKill",
        "bounds": bounds,
        "budget": budget,
        "parameters": parameters,
        "batch_size": batch_size,
        "kill_after": kill_after,  # kills when sample 4 is done
    }
    runner_config = {"type": "ExampleRunner"}

    base_run_dir = tmp_path

    args = SimpleNamespace(
        executors={"e1": executor_config},
        samplers={"s1": sampler_config},
        runners={"r1": runner_config},
        supervisor={
            "run_mode": "fresh",
            "base_run_dir": base_run_dir,
            "run_order": [{"executor": "e1", "sampler": "s1", "runner": "r1"}],
        },
    )

    supervisor = Supervisor(args)
    try:
        supervisor.start()
        assert False  # This shouldn't happen
    except Exception:
        pass  # This should happen

    assert len(next(os.walk(tmp_path))[1]) == kill_after * batch_size

    args.supervisor["run_mode"] = "resume"  # continue where the previous one left off
    args.samplers["s1"]["type"] = "RandomSamplerWithKill"
    args.samplers["s1"]["kill_after"] = None  # don't kill the poor sampler

    supervisor2 = Supervisor(args)
    supervisor2.start()

    assert len(next(os.walk(tmp_path))[1]) == budget


def test_full_workflow_dask(tmp_path):
    config = {}
    # -- sampler
    # TODO: test different samplers
    bounds = [[-5, 5], [0, 1]]
    parameters = ["c1", "c2"]
    budget = 50
    executor_config = {
        "type": "DaskExecutor",
        "LocalCluster_config": {
            "name": "es-dask_cluster",
            "n_workers": 2,
            "threads_per_worker": 1,
        },
    }
    sampler_config = {
        "type": "RandomSampler",
        "bounds": bounds,
        "budget": budget,
        "parameters": parameters,
    }
    runner_config = {"type": "ExampleRunner"}

    base_run_dir = tmp_path

    args = SimpleNamespace(
        executors={"e1": executor_config},
        samplers={"s1": sampler_config},
        runners={"r1": runner_config},
        supervisor={
            "base_run_dir": base_run_dir,
            "run_order": [{"executor": "e1", "sampler": "s1", "runner": "r1"}],
        },
    )

    supervisor = Supervisor(args)
    supervisor.start()
    # This should create {budget} folders with ??? inside

    assert len(next(os.walk(tmp_path))[1]) == budget
