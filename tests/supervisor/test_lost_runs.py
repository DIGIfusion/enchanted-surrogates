# pylint: disable=E1101
"""Lost run detection: the supervisor stops waiting for runs whose task disappeared."""

import os
import threading
import time as _time
from types import SimpleNamespace as _NS

import pandas as pd

from enchanted_surrogates.supervisor.supervisor import Supervisor


def make_args(tmp_path):
    """Constructor arguments of a Supervisor with a single mock level."""
    return _NS(
        executors={"testexecutor": {"type": "mock"}},
        samplers={"testsampler": {"type": "mock"}},
        runners={"testrunner": {"type": "mock"}},
        supervisor={
            "base_run_dir": str(tmp_path),
            "summary_datatype": "csv",
            "run_order": [{"executor": "testexecutor", "sampler": "testsampler", "runner": "testrunner"}],
            "save_files_list": ["keep_me.txt"],
        },
        runner={"type": "mock"},
        storage={"type": "mock"},
    )


# ---- lost runs ---------------------------------------------------------------------------------------------------------

def _monitor_in_thread(supervisor, run_dirs, executor, samples=None, timeout=30):
    """Run monitor_runs in a thread so that a monitor that waits forever fails the test and does not hang it."""
    runner = supervisor.nested_groups[0].runners[0]
    errors = []

    def target():
        try:
            supervisor.monitor_runs(
                group_name="G0", runner_config=runner, run_dirs=run_dirs, nested_depth=0, sequential_depth=0,
                batch_number=0, group_start_time=_time.time(), executor=executor, samples=samples)
        except BaseException as exc:   # a crashed monitor must fail the test, not look like a finished one
            errors.append(exc)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if errors:
        raise errors[0]
    return not t.is_alive()


def _supervisor(tmp_path, patch_supervisor_imports, **options):
    patch_supervisor_imports()
    args = make_args(tmp_path)
    args.supervisor.update(options)
    return Supervisor(args)


def _write_result(run_dir, **row):
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame({k: [v] for k, v in {"success": True, **row}.items()}).to_csv(
        os.path.join(run_dir, "enchanted_datapoint.csv"), index=False)


def test_a_lost_run_is_recorded_as_failed_when_the_executor_has_no_task_left(tmp_path, patch_supervisor_imports):
    supervisor = _supervisor(tmp_path, patch_supervisor_imports, lost_run_grace=0)
    done, lost = str(tmp_path / "data" / "r_done"), str(tmp_path / "data" / "r_lost")
    _write_result(done, a=0)
    os.makedirs(lost)
    executor = _NS(is_tasks_running=lambda: False)

    assert _monitor_in_thread(supervisor, [done, lost], executor, samples={lost: {"a": 1, "c": 9, "success": True}})

    row = pd.read_csv(os.path.join(lost, "enchanted_datapoint.csv")).iloc[0]
    assert row["success"] == False                       # noqa: E712
    assert "Run lost" in row["error"]
    assert row["a"] == 1 and row["c"] == 9               # the parameters of the sample are kept
    assert row["run_dir"] == lost
    assert pd.read_csv(os.path.join(done, "enchanted_datapoint.csv")).iloc[0]["success"]   # the finished run is untouched


def test_nothing_is_declared_lost_while_the_executor_still_has_tasks(tmp_path, patch_supervisor_imports):
    supervisor = _supervisor(tmp_path, patch_supervisor_imports, lost_run_grace=0)
    run = str(tmp_path / "data" / "r0")
    os.makedirs(run)
    threading.Timer(2.5, _write_result, args=(run,), kwargs=dict(a=5)).start()

    assert _monitor_in_thread(supervisor, [run], _NS(is_tasks_running=lambda: True))
    row = pd.read_csv(os.path.join(run, "enchanted_datapoint.csv")).iloc[0]
    assert row["success"] and row["a"] == 5              # the real result, not a lost-run row


def test_a_result_that_arrives_within_the_grace_period_is_not_a_lost_run(tmp_path, patch_supervisor_imports):
    supervisor = _supervisor(tmp_path, patch_supervisor_imports, lost_run_grace=6)
    run = str(tmp_path / "data" / "r0")
    os.makedirs(run)
    threading.Timer(2.0, _write_result, args=(run,), kwargs=dict(a=5)).start()

    assert _monitor_in_thread(supervisor, [run], _NS(is_tasks_running=lambda: False))
    row = pd.read_csv(os.path.join(run, "enchanted_datapoint.csv")).iloc[0]
    assert row["success"] and row["a"] == 5


def test_lost_run_detection_can_be_switched_off(tmp_path, patch_supervisor_imports):
    supervisor = _supervisor(tmp_path, patch_supervisor_imports, lost_run_grace=0, detect_lost_runs=False)
    run = str(tmp_path / "data" / "r0")
    os.makedirs(run)
    threading.Timer(2.5, _write_result, args=(run,), kwargs=dict(a=5)).start()

    assert _monitor_in_thread(supervisor, [run], _NS(is_tasks_running=lambda: False))
    assert pd.read_csv(os.path.join(run, "enchanted_datapoint.csv")).iloc[0]["a"] == 5


def test_monitor_without_an_executor_behaves_as_before(tmp_path, patch_supervisor_imports):
    supervisor = _supervisor(tmp_path, patch_supervisor_imports, lost_run_grace=0)
    run = str(tmp_path / "data" / "r0")
    os.makedirs(run)
    threading.Timer(2.0, _write_result, args=(run,), kwargs=dict(a=5)).start()
    assert _monitor_in_thread(supervisor, [run], None)


def test_start_records_a_lost_run_and_goes_on(tmp_path, patch_supervisor_imports):
    args = make_args(tmp_path)
    args.supervisor["lost_run_grace"] = 0
    samplers, executors = patch_supervisor_imports([[[{"a": 0}, {"a": 1}, {"a": 2}]]])
    supervisor = Supervisor(args)          # creates the mock executor
    executor = executors[0]
    real_execute = executor.execute.side_effect

    def execute_but_lose_one(input, runner_config):
        real_execute([(d, s) for d, s in input if s["a"] != 1], runner_config)   # the task of a == 1 never writes anything
        for d, s in input:
            os.makedirs(d, exist_ok=True)
    executor.execute.side_effect = execute_but_lose_one
    executor.is_tasks_running.return_value = False

    done = []
    t = threading.Thread(target=lambda: (supervisor.start(), done.append(True)), daemon=True)
    t.start()
    t.join(60)
    assert done, "start() did not finish: it waited for the lost run"

    rows = pd.concat([pd.read_csv(f) for f in sorted((tmp_path / "data").glob("*/enchanted_datapoint.csv"))])
    assert len(rows) == 3
    assert rows.set_index("a")["success"].to_dict() == {0: True, 1: False, 2: True}
