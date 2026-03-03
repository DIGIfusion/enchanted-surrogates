"""
Full workflow tests for seamless sampling continuation.
"""

import os

from types import SimpleNamespace
from enchanted_surrogates.supervisor.supervisor import Supervisor

executor_config = {"type": "LocalExecutor"}
runner_config = {"type": "ExampleRunner"}

def test_simple_resume_after_interruption(tmp_path):
    bounds = [[-5, 5], [0, 1]]
    parameters = ["c1", "c2"]
    budget = 50
    kill_after = 4
    batch_size = 10
    sampler_config = {
        "type": "RandomSamplerWithKill",
        "bounds": bounds,
        "budget": budget,
        "parameters": parameters,
        "batch_size": batch_size,
        "kill_after": kill_after,  # kills when sample 4 is done
    }

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
    args.samplers["s1"]["kill_after"] = None  # don't kill the poor sampler

    supervisor2 = Supervisor(args)
    supervisor2.start()

    assert len(next(os.walk(tmp_path))[1]) == budget

def test_nested_resume_after_interruption(tmp_path):
    bounds = [[-5, 5], [0, 1]]
    budget = 9
    batch_size = 3
    kill_after = 2
    sampler_config_first = {
        "type": "RandomSampler",
        "bounds": bounds,
        "budget": budget,
        "parameters": ["c1", "c2"],
        "batch_size": batch_size
    }
    sampler_config_second = {
        "type": "RandomSamplerWithKill",
        "bounds": bounds,
        "budget": budget,
        "parameters": ["c3", "c4"],
        "batch_size": batch_size,
        "kill_after": kill_after
    }

    base_run_dir = tmp_path

    args = SimpleNamespace(
        executors={"e1": executor_config},
        samplers={"s1": sampler_config_first, "s2": sampler_config_second},
        runners={"r1": runner_config},
        supervisor={
            "run_mode": "fresh",
            "base_run_dir": base_run_dir,
            "run_order": [
                {"executor": "e1", "sampler": "s1", "runner": "r1"},
                {"executor": "e1", "sampler": "s2", "runner": "r1"}
            ],
        },
    )

    supervisor = Supervisor(args)
    try:
        supervisor.start()
        assert False  # This shouldn't happen
    except Exception:
        pass  # This should happen

    assert len(next(os.walk(tmp_path))[1]) == budget + budget * kill_after * batch_size

    args.supervisor["run_mode"] = "resume"  # continue where the previous one left off
    args.samplers["s2"]["kill_after"] = None  # don't kill the poor sampler

    supervisor2 = Supervisor(args)
    supervisor2.start()

    assert len(next(os.walk(tmp_path))[1]) == budget + budget * budget

def test_simple_resume_with_increased_budget(tmp_path):
    old_budget = 15
    new_budget = 25

    sampler_config = {
        "type": "RandomSampler",
        "bounds": [[-5, 5], [0, 1]],
        "budget": old_budget,
        "parameters": ["c1", "c2"],
        "batch_size": 5
    }

    args = SimpleNamespace(
        executors={"e1": executor_config},
        samplers={"s1": sampler_config},
        runners={"r1": runner_config},
        supervisor={
            "run_mode": "fresh",
            "base_run_dir": tmp_path,
            "run_order": [{"executor": "e1", "sampler": "s1", "runner": "r1"}],
        }
    )

    # Run until the end
    supervisor = Supervisor(args)
    supervisor.start()
    assert len(next(os.walk(tmp_path))[1]) == old_budget

    # Increase budget and re-run with resume, (new_budget - old_budget) new samples should be got
    args.supervisor["run_mode"] = "resume"
    args.samplers["s1"]["budget"] = new_budget  # increase budget

    supervisor2 = Supervisor(args)
    supervisor2.start()

    assert len(next(os.walk(tmp_path))[1]) == new_budget

def test_simple_extend(tmp_path):
    old_budget = 15
    extend_budget_by = 10

    sampler_config = {
        "type": "RandomSampler",
        "bounds": [[-5, 5], [0, 1]],
        "budget": old_budget,
        "parameters": ["c1", "c2"],
        "batch_size": 5
    }

    args = SimpleNamespace(
        executors={"e1": executor_config},
        samplers={"s1": sampler_config},
        runners={"r1": runner_config},
        supervisor={
            "run_mode": "fresh",
            "base_run_dir": tmp_path,
            "run_order": [{"executor": "e1", "sampler": "s1", "runner": "r1"}],
        }
    )

    # Run until the end
    supervisor = Supervisor(args)
    supervisor.start()
    assert len(next(os.walk(tmp_path))[1]) == old_budget

    # Set budget and re-run with extend, (extend_budget_by) new samples should be got
    args.supervisor["run_mode"] = "extend"
    args.samplers["s1"]["budget"] = extend_budget_by  # increase budget

    supervisor2 = Supervisor(args)
    supervisor2.start()

    assert len(next(os.walk(tmp_path))[1]) == old_budget + extend_budget_by

def test_nested_extend(tmp_path):
    first_budget = 9
    second_budget = 9
    batch_size = 3
    extend_budget_by = 3

    sampler_config_first = {
        "type": "RandomSampler",
        "bounds": [[-5, 5], [0, 1]],
        "budget": first_budget,
        "parameters": ["c1", "c2"],
        "batch_size": batch_size
    }
    sampler_config_second = {
        "type": "RandomSampler",
        "bounds": [[-5, 5], [0, 1]],
        "budget": second_budget,
        "parameters": ["c1", "c2"],
        "batch_size": batch_size
    }

    args = SimpleNamespace(
        executors={"e1": executor_config},
        samplers={"s1": sampler_config_first, "s2": sampler_config_second},
        runners={"r1": runner_config},
        supervisor={
            "run_mode": "fresh",
            "base_run_dir": tmp_path,
            "run_order": [
                {"executor": "e1", "sampler": "s1", "runner": "r1"},
                {"executor": "e1", "sampler": "s2", "runner": "r1"}
            ]
        }
    )

    # Run until the end
    supervisor = Supervisor(args)
    supervisor.start()
    assert len(next(os.walk(tmp_path))[1]) == first_budget + first_budget * second_budget

    # Set budget and re-run with extend, (extend_budget_by) new samples should be got
    args.supervisor["run_mode"] = "extend"
    args.samplers["s2"]["budget"] = extend_budget_by

    supervisor2 = Supervisor(args)
    supervisor2.start()

    assert len(next(os.walk(tmp_path))[1]) == (
        first_budget + first_budget * (second_budget + extend_budget_by)
    )
