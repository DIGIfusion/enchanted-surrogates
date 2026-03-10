import os
from enchanted_surrogates.supervisor.supervisor import Supervisor
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize("batch_count_a", [1, 2, 7])
@pytest.mark.parametrize("batch_count_b", [1, 2, 4])
def test_supervisor_batches(batch_count_a, batch_count_b, patch_supervisor_imports, tmp_path):
    args = make_nested_args(tmp_path, nested_count = 2)

    samplers, executors = patch_supervisor_imports([
        [ # sampler a
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}] for i in range(0, batch_count_a)
        ],
        [ # sampler b
            [{"c": 5, "d": 6}, {"c": 7, "d": 8}] for i in range(0, batch_count_b)
        ]
    ])

    # run code under test
    supervisor = Supervisor(args)
    supervisor.start()

    assert samplers[0].get_next_samples.call_count == batch_count_a
    assert samplers[1].get_next_samples.call_count == batch_count_b
    assert executors[0].execute.call_count == batch_count_a
    assert executors[1].execute.call_count == batch_count_b
    assert len(next(os.walk(tmp_path / "data"))[1]) == 2 * batch_count_a + 2 * batch_count_a * 2 * batch_count_b

def make_nested_args(tmp_path, nested_count: int):
    """
    Helper function to create constructor arguments with nested levels.
    Each nesting level gets its own executor, sampler, and runner.
    """
    executors = {}
    samplers = {}
    runners = {}
    run_order = []

    for i in range(nested_count):
        executor_name = f"testexecutor_{i}"
        sampler_name = f"testsampler_{i}"
        runner_name = f"testrunner_{i}"

        executors[executor_name] = {"type": "mock"}
        samplers[sampler_name] = {"type": "mock"}
        runners[runner_name] = {"type": "mock"}

        run_order.append({
            "executor": executor_name,
            "sampler": sampler_name,
            "runner": runner_name,
        })

    return SimpleNamespace(
        executors=executors,
        samplers=samplers,
        runners=runners,
        supervisor={
            "base_run_dir": str(tmp_path),
            "run_order": run_order
        }
    )
