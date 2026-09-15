# pylint: disable=E1101

import os
import pandas as pd
import h5py
from enchanted_surrogates.supervisor.supervisor import Supervisor
from types import SimpleNamespace


def test_all_processes_done_returns_correct_values(tmp_path, patch_supervisor_imports):
    patch_supervisor_imports()
    supervisor = Supervisor(make_args(tmp_path))

    one_run_dir = tmp_path / "data" / "d0_b0_s0_r0"
    one_run_dir.mkdir()
    assert supervisor.all_processes_done() is False

    pd.DataFrame([{"x": 1}]).to_csv(
        one_run_dir / "enchanted_datapoint.csv", index=False
    )
    assert supervisor.all_processes_done() is True


def test_create_dataset_combines_csv_files(tmp_path, patch_supervisor_imports):
    patch_supervisor_imports()
    supervisor = Supervisor(make_args(tmp_path))

    data_path = tmp_path / "data"
    create_run_folders(data_path, 3)

    df = supervisor.create_dataset()
    assert len(df) == 3
    assert set(df["x"]) == {0, 1, 2}


def test_create_hdf5_storage_format(tmp_path, patch_supervisor_imports):
    patch_supervisor_imports()
    supervisor = Supervisor(make_args(tmp_path))

    data_path = tmp_path / "data"
    run_folders = create_run_folders(data_path, 3)

    df = supervisor.create_dataset()

    supervisor.hdf5_append_datapoints(run_folders)
    supervisor.hdf5_write_aggregate_dataset_and_metadata(df)

    # Check if hdf5 exists
    output_file = tmp_path / "runs.h5"
    assert output_file.exists()
    assert output_file.is_file()

    with h5py.File(output_file, "r") as file:
        # Then, check structure
        assert "data" in file
        assert "data/aggregated" in file
        assert "data/runs" in file
        assert "metadata" in file

        # Check groups
        group = file["data/aggregated"]
        assert "values" in group
        assert "columns" in group

        # Check aggregation dimensions and values
        agg_values = file["data/aggregated/values"][:]
        agg_columns = file["data/aggregated/columns"][:]

        assert agg_values.shape == (3, 1)
        assert sorted(agg_values.flatten().tolist()) == [0, 1, 2]
        assert agg_columns.tolist() == [b"x"]

        # Check run dimensions and values
        run_values = file["data/runs/d0_b2_s0_r0/values"][:]
        run_columns = file["data/runs/d0_b2_s0_r0/columns"][:]

        assert run_values.shape == (1, 1)
        assert run_values[0, 0] == 2
        assert run_columns.tolist() == [b"x"]

        # Check that metadata exists
        meta = file["metadata/run_groups/0"].attrs
        for key in ["executors", "sampler", "runners"]:
            assert key in meta


def test_start_calls_execute_for_each_sample(tmp_path, patch_supervisor_imports):
    samplers, executors = patch_supervisor_imports(
        [
            [  # sampler 1
                [{"a": 1}, {"a": 2}],
                [{"a": 3}],
            ]
        ]
    )

    supervisor = Supervisor(make_args(tmp_path))
    supervisor.start()

    assert samplers[-1].get_next_samples.call_count == 2
    assert executors[-1].execute.call_count == 2


def test_start_cleans_executors_only_when_no_longer_needed(
    tmp_path, patch_supervisor_imports
):
    """
    Nested + sequential + batched scenario:
      - depth 0 runs two executors sequentially (a, then b) and has two batches
      - depth 1 runs a single executor (c) and has one batch

    An executor must never be cleaned while it is still going to be reused
    (e.g. executor a/b between depth 0's two batches), and every executor
    must eventually be cleaned once its group's sampler has exhausted its
    budget.
    """
    samplers, executors = patch_supervisor_imports(
        [
            [  # depth 0 sampler: two batches
                [{"a": 1}],
                [{"a": 2}],
            ],
            [  # depth 1 sampler: one batch
                [{"a": 3}],
            ],
        ]
    )

    args = make_args(tmp_path)
    args.executors = {
        "executor_0a": {"type": "mock"},
        "executor_0b": {"type": "mock"},
        "executor_1": {"type": "mock"},
    }
    args.samplers = {
        "sampler_0": {"type": "mock"},
        "sampler_1": {"type": "mock"},
    }
    args.runners = {
        "runner_0a": {"type": "mock"},
        "runner_0b": {"type": "mock"},
        "runner_1": {"type": "mock"},
    }
    args.supervisor["run_order"] = [
        {
            "executor": ["executor_0a", "executor_0b"],
            "sampler": "sampler_0",
            "runner": ["runner_0a", "runner_0b"],
        },
        {
            "executor": "executor_1",
            "sampler": "sampler_1",
            "runner": "runner_1",
        },
    ]

    supervisor = Supervisor(args)

    # `executors` is only populated once Supervisor.__init__ imports them above.
    exec_a, exec_b, exec_c = executors
    names = {id(exec_a): "a", id(exec_b): "b", id(exec_c): "c"}
    events = []

    for executor in executors:
        name = names[id(executor)]
        original_execute = executor.execute.side_effect

        def _execute(input, runner_config, _orig=original_execute, _name=name):
            _orig(input, runner_config)
            events.append((_name, "execute"))

        executor.execute.side_effect = _execute
        executor.clean.side_effect = lambda *a, _name=name, **k: events.append(
            (_name, "clean")
        )

    supervisor.start()

    def kinds(name):
        return [kind for n, kind in events if n == name]

    # executor a/b are each used once per depth-0 batch, executor c once overall
    assert exec_a.execute.call_count == 2
    assert exec_b.execute.call_count == 2
    assert exec_c.execute.call_count == 1

    # Every executor is eventually cleaned up.
    assert exec_a.clean.called
    assert exec_b.clean.called
    assert exec_c.clean.called

    # Critically: a/b must not be cleaned in between depth 0's two batches -
    # both executes must happen before the first clean for that executor.
    for name in ("a", "b"):
        executor_kinds = kinds(name)
        first_clean = executor_kinds.index("clean")
        assert executor_kinds[:first_clean].count("execute") == 2, (
            f"executor {name} was cleaned before finishing both of its batches: "
            f"{executor_kinds}"
        )


def test_start_keeps_executor_shared_across_depths_open(
    tmp_path, patch_supervisor_imports
):
    """
    If the same executor (by name) is used by both depth 0 and depth 1, it must
    stay open across the depth boundary - it should only be cleaned once every
    batch at every depth that uses it has finished, not as soon as depth 0
    (which happens first) exhausts its own budget.
    """
    samplers, executors = patch_supervisor_imports(
        [
            [  # depth 0 sampler: two batches
                [{"a": 1}],
                [{"a": 2}],
            ],
            [  # depth 1 sampler: one batch
                [{"a": 3}],
            ],
        ]
    )

    args = make_args(tmp_path)
    args.executors = {"shared": {"type": "mock"}}
    args.samplers = {
        "sampler_0": {"type": "mock"},
        "sampler_1": {"type": "mock"},
    }
    args.runners = {
        "runner_0": {"type": "mock"},
        "runner_1": {"type": "mock"},
    }
    args.supervisor["run_order"] = [
        {"executor": "shared", "sampler": "sampler_0", "runner": "runner_0"},
        {"executor": "shared", "sampler": "sampler_1", "runner": "runner_1"},
    ]

    supervisor = Supervisor(args)

    (shared_executor,) = executors
    events = []
    original_execute = shared_executor.execute.side_effect

    def _execute(input, runner_config):
        original_execute(input, runner_config)
        events.append("execute")

    shared_executor.execute.side_effect = _execute
    shared_executor.clean.side_effect = lambda *a, **k: events.append("clean")

    supervisor.start()

    # used once per depth-0 batch (2) plus once for depth 1's single batch
    assert shared_executor.execute.call_count == 3

    first_clean = events.index("clean")
    assert events[:first_clean].count("execute") == 3, (
        "shared executor was cleaned before depth 1 (which still needed it) "
        f"had run: {events}"
    )


def test_save_files_option_all(tmp_path, patch_supervisor_imports):
    patch_supervisor_imports()
    supervisor = Supervisor(make_args(tmp_path))
    create_dummy_files(path=tmp_path)
    supervisor.delete_unwanted_files(argument="all")

    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "enchanted_dataset.csv").exists()
    assert (tmp_path / "runs.h5").exists()

    assert (tmp_path / "subfolder" / "subfile.txt").exists()
    assert (tmp_path / "subfolder" / "keep_me.txt").exists()


def test_save_files_option_custom(tmp_path, patch_supervisor_imports):
    patch_supervisor_imports()
    supervisor = Supervisor(make_args(tmp_path))
    create_dummy_files(path=tmp_path)
    supervisor.delete_unwanted_files("custom")
    assert not (tmp_path / "config.yaml").exists()
    assert (tmp_path / "enchanted_dataset.csv").exists()
    assert (tmp_path / "runs.h5").exists()

    assert not (tmp_path / "subfolder" / "subfile.txt").exists()
    assert (tmp_path / "subfolder" / "keep_me.txt").exists()


def test_save_files_option_none(tmp_path, patch_supervisor_imports):
    patch_supervisor_imports()
    supervisor = Supervisor(make_args(tmp_path))
    create_dummy_files(path=tmp_path)
    supervisor.delete_unwanted_files("none")
    assert not (tmp_path / "config.yaml").exists()
    assert (tmp_path / "enchanted_dataset.csv").exists()
    assert (tmp_path / "runs.h5").exists()

    assert not (tmp_path / "subfolder" / "subfile.txt").exists()
    assert not (tmp_path / "subfolder" / "keep_me.txt").exists()


def test_local_storage(tmp_path, patch_supervisor_imports):
    patch_supervisor_imports()

    run_dir = os.path.join(tmp_path, "run_dir")
    fast_dir = os.path.join(tmp_path, "fast_dir")

    os.makedirs(run_dir)
    os.makedirs(fast_dir)

    supervisor = Supervisor(make_args(run_dir, local_storage=fast_dir))

    assert supervisor.local_storage == fast_dir

    os.rmdir(os.path.join(run_dir, "data"))

    os.environ["abcdef123456"] = fast_dir
    supervisor = Supervisor(make_args(run_dir, local_storage="abcdef123456"))

    assert supervisor.local_storage == fast_dir


def make_args(tmp_path, summary="csv", local_storage=None):
    """
    Helper function to create constructor arguments
    """
    supervisor = {
        "base_run_dir": str(tmp_path),
        "summary_datatype": summary,
        "run_order": [
            {
                "executor": "testexecutor",
                "sampler": "testsampler",
                "runner": "testrunner",
            }
        ],
        "save_files_list": ["keep_me.txt"],
    }

    if local_storage is not None:
        supervisor["local_storage"] = local_storage

    return SimpleNamespace(
        executors={"testexecutor": {"type": "mock"}},
        samplers={"testsampler": {"type": "mock"}},
        runners={"testrunner": {"type": "mock"}},
        supervisor=supervisor,
        runner={"type": "mock"},
        storage={"type": "mock"},
    )


def create_run_folders(tmp_path, amount) -> list[str]:
    """
    Helper function to create run folders
    """
    folders = []

    for i in range(amount):
        d = tmp_path / f"d0_b{i}_s0_r0"
        folders.append(d)
        d.mkdir()
        pd.DataFrame([{"x": i}]).to_csv(d / "enchanted_datapoint.csv", index=False)

    return folders


def create_dummy_files(path):
    subfolder = path / "subfolder"
    subfolder.mkdir()

    # Files in the root
    (path / "config.yaml").write_text("dummy")
    (path / "enchanted_dataset.csv").write_text("dummy")
    (path / "runs.h5").write_text("dummy")

    # Files in the subfolder
    (subfolder / "subfile.txt").write_text("dummy")
    (subfolder / "keep_me.txt").write_text("dummy")

