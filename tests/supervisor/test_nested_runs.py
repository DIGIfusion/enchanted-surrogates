import os
import glob
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


def test_earlier_nested_group_files_kept_until_later_group_done(patch_supervisor_imports, tmp_path):
    """
    An earlier nested group's (e.g. HELENA's) intermediate_output.dat files should not
    be deleted until the later nested group (e.g. GENE) that reads them has finished,
    even though save_files="none" would otherwise delete them as soon as their own
    group's batches complete.
    """
    args = make_nested_args(tmp_path, nested_count=2, save_files="none")

    patch_supervisor_imports([
        [ # sampler a
            [{"a": 1, "b": 2}],
        ],
        [ # sampler b
            [{"c": 5, "d": 6}],
        ],
    ])

    supervisor = Supervisor(args)

    original_monitor_runs = supervisor.monitor_runs
    marker_existed_when_next_group_started = {}

    def spying_monitor_runs(*args, **kwargs):
        nested_depth = kwargs.get("nested_depth")
        if nested_depth is not None and nested_depth > 0:
            previous_dirs = glob.glob(str(tmp_path / "data" / f"dn{nested_depth - 1}_ds0_b0_s*"))
            marker_existed_when_next_group_started[nested_depth - 1] = bool(previous_dirs) and all(
                os.path.exists(os.path.join(d, "intermediate_output.dat")) for d in previous_dirs
            )
        return original_monitor_runs(*args, **kwargs)

    supervisor.monitor_runs = spying_monitor_runs
    supervisor.start()

    assert marker_existed_when_next_group_started.get(0) is True, (
        "nested_depth=0's intermediate_output.dat was deleted before nested_depth=1 started"
    )


def test_delete_immediately_deletes_before_next_group_starts(patch_supervisor_imports, tmp_path):
    """
    With delete_immediately=True, the old eager-cleanup behavior is restored: an earlier
    nested group's files are deleted as soon as its own runs finish, not deferred.
    """
    args = make_nested_args(tmp_path, nested_count=2, save_files="none", delete_immediately=True)

    patch_supervisor_imports([
        [ # sampler a
            [{"a": 1, "b": 2}],
        ],
        [ # sampler b
            [{"c": 5, "d": 6}],
        ],
    ])

    supervisor = Supervisor(args)

    original_monitor_runs = supervisor.monitor_runs
    marker_existed_when_next_group_started = {}

    def spying_monitor_runs(*args, **kwargs):
        nested_depth = kwargs.get("nested_depth")
        if nested_depth is not None and nested_depth > 0:
            previous_dirs = glob.glob(str(tmp_path / "data" / f"dn{nested_depth - 1}_ds0_b0_s*"))
            marker_existed_when_next_group_started[nested_depth - 1] = bool(previous_dirs) and any(
                os.path.exists(os.path.join(d, "intermediate_output.dat")) for d in previous_dirs
            )
        return original_monitor_runs(*args, **kwargs)

    supervisor.monitor_runs = spying_monitor_runs
    supervisor.start()

    assert marker_existed_when_next_group_started.get(0) is False


def make_nested_args(tmp_path, nested_count: int, save_files: str | None = None, delete_immediately: bool | None = None):
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

    supervisor_config = {
        "base_run_dir": str(tmp_path),
        "run_order": run_order
    }
    if save_files is not None:
        supervisor_config["save_files"] = save_files
    if delete_immediately is not None:
        supervisor_config["delete_immediately"] = delete_immediately

    return SimpleNamespace(
        executors=executors,
        samplers=samplers,
        runners=runners,
        supervisor=supervisor_config
    )
