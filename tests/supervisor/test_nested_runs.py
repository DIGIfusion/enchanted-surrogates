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

def make_nested_args(tmp_path, nested_count: int, supervisor_options: dict | None = None):
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
            "run_order": run_order,
            **(supervisor_options or {}),
        }
    )


def _run_two_level(tmp_path, patch_supervisor_imports, parents_success, skip, n_children=3):
    """Two nested levels: level 0 samples with the given success values, level 1 with n_children samples."""
    args = make_nested_args(tmp_path, nested_count=2, supervisor_options={"skip_failed_parents": skip})
    samplers, executors = patch_supervisor_imports([
        [[{"a": i, "success": ok} for i, ok in enumerate(parents_success)]],
        [[{"c": j} for j in range(n_children)]],
    ])
    supervisor = Supervisor(args)
    supervisor.start()
    return samplers, executors


def _n_run_dirs(tmp_path, level):
    return len([d for d in os.listdir(tmp_path / "data") if d.startswith(f"dn{level}_")])


def test_failed_parents_get_children_by_default(patch_supervisor_imports, tmp_path):
    _run_two_level(tmp_path, patch_supervisor_imports, [True, False, True, False], skip=False)
    assert _n_run_dirs(tmp_path, 0) == 4
    assert _n_run_dirs(tmp_path, 1) == 4 * 3


def test_skip_failed_parents_runs_children_of_successful_parents_only(patch_supervisor_imports, tmp_path):
    _, executors = _run_two_level(tmp_path, patch_supervisor_imports, [True, False, True, False], skip=True)
    assert _n_run_dirs(tmp_path, 0) == 4
    assert _n_run_dirs(tmp_path, 1) == 2 * 3
    # the children carry the parameters of the successful parents only
    children = [sample for _, sample in executors[1].execute.call_args[0][0]]
    assert sorted({c["a"] for c in children}) == [0, 2]
    assert all(c["success"] is True or c["success"] == True for c in children)


def test_skip_failed_parents_when_all_parents_failed(patch_supervisor_imports, tmp_path):
    _, executors = _run_two_level(tmp_path, patch_supervisor_imports, [False, False], skip=True)
    assert _n_run_dirs(tmp_path, 0) == 2
    assert _n_run_dirs(tmp_path, 1) == 0
    assert executors[1].execute.call_count == 0


def test_skip_failed_parents_does_not_change_the_first_level(patch_supervisor_imports, tmp_path):
    # success False in the level 0 samples must not stop level 0 itself
    _run_two_level(tmp_path, patch_supervisor_imports, [False, False, False], skip=True, n_children=1)
    assert _n_run_dirs(tmp_path, 0) == 3
