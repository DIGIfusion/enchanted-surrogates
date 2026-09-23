import os
import glob
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


@pytest.mark.parametrize("sequence_count", [2, 3, 5])
def test_intermediate_stage_files_kept_until_next_stage_done(sequence_count, patch_supervisor_imports, tmp_path):
    """
    Intermediate-output.dat files (standing in for e.g. HELENA's eliteinp, which a later
    GENE stage needs to read) should not be deleted until the NEXT sequential stage for
    that sample has finished, even though save_files="none" would otherwise delete them
    as soon as their own stage completes.
    """
    args = make_sequential_args(tmp_path, sequence_count, save_files="none")

    patch_supervisor_imports([
        [ # sampler
            [{"a": 1, "b": 2}],
        ]
    ])

    supervisor = Supervisor(args)

    original_monitor_runs = supervisor.monitor_runs
    marker_existed_when_next_stage_started = {}

    def spying_monitor_runs(*args, **kwargs):
        sequential_depth = kwargs.get("sequential_depth")
        if sequential_depth is not None and sequential_depth > 0:
            previous_dirs = glob.glob(str(tmp_path / "data" / f"dn0_ds{sequential_depth - 1}_b0_s*"))
            marker_existed_when_next_stage_started[sequential_depth - 1] = bool(previous_dirs) and all(
                os.path.exists(os.path.join(d, "intermediate_output.dat")) for d in previous_dirs
            )
        return original_monitor_runs(*args, **kwargs)

    supervisor.monitor_runs = spying_monitor_runs
    supervisor.start()

    # Every non-final stage's marker file must still have existed once the following
    # stage started monitoring (i.e. it was not deleted prematurely).
    for sequential_depth in range(sequence_count - 1):
        assert marker_existed_when_next_stage_started.get(sequential_depth) is True, (
            f"stage {sequential_depth}'s intermediate_output.dat was deleted before stage {sequential_depth + 1} started"
        )


def test_delete_immediately_deletes_before_next_stage_starts(patch_supervisor_imports, tmp_path):
    """
    With delete_immediately=True, the old eager-cleanup behavior is restored: an earlier
    sequential stage's files are deleted as soon as its own runs finish, not deferred.
    """
    args = make_sequential_args(tmp_path, sequence_count=2, save_files="none", delete_immediately=True)

    patch_supervisor_imports([
        [ # sampler
            [{"a": 1, "b": 2}],
        ]
    ])

    supervisor = Supervisor(args)

    original_monitor_runs = supervisor.monitor_runs
    marker_existed_when_next_stage_started = {}

    def spying_monitor_runs(*args, **kwargs):
        sequential_depth = kwargs.get("sequential_depth")
        if sequential_depth is not None and sequential_depth > 0:
            previous_dirs = glob.glob(str(tmp_path / "data" / f"dn0_ds{sequential_depth - 1}_b0_s*"))
            marker_existed_when_next_stage_started[sequential_depth - 1] = bool(previous_dirs) and any(
                os.path.exists(os.path.join(d, "intermediate_output.dat")) for d in previous_dirs
            )
        return original_monitor_runs(*args, **kwargs)

    supervisor.monitor_runs = spying_monitor_runs
    supervisor.start()

    assert marker_existed_when_next_stage_started.get(0) is False


def make_sequential_args(tmp_path, sequence_count: int, save_files: str | None = None, delete_immediately: bool | None = None):
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
