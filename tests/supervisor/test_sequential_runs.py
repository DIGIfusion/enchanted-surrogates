import os
from enchanted_surrogates.supervisor.supervisor import Supervisor
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize("sequence_count", [2, 3, 15])
def test_supervisor_batches(sequence_count, patch_supervisor_imports, tmp_path):
    args = make_sequential_args(tmp_path, sequence_count)
    batch_count = 4

    samplers, executors = patch_supervisor_imports([
        [ # sampler
            [{"a": 1, "b": 2}] for i in range(0, batch_count)
        ]
    ])

    supervisor = Supervisor(args)
    supervisor.start()

    assert samplers[0].get_next_samples.call_count == batch_count
    assert sum(executor.execute.call_count for executor in executors) == batch_count * sequence_count
    assert len(next(os.walk(tmp_path / "data"))[1]) == sequence_count * batch_count

def make_sequential_args(tmp_path, sequence_count: int):
    """
    Helper function to create constructor arguments with multiple sequential runners.
    One sampler is used and sequence_count specifies how many executors and runners there will be.
    """
    executors = {}
    runners = {}

    sampler_name = "testsampler"
    samplers = {
        sampler_name: {
            "type": "mock"
        }
    }
    run_order = [{
        "sampler": sampler_name,
        "executor": [],
        "runner": []
    }]

    for i in range(sequence_count):
        executor_name = f"testexecutor_{i}"
        runner_name = f"testrunner_{i}"

        executors[executor_name] = {"type": "mock"}
        runners[runner_name] = {"type": "mock"}

        run_order[0]["executor"].append(
            executor_name
        )
        run_order[0]["runner"].append(
            runner_name
        )

    return SimpleNamespace(
        executors=executors,
        samplers=samplers,
        runners=runners,
        supervisor={
            "base_run_dir": str(tmp_path),
            "run_order": run_order
        }
    )
