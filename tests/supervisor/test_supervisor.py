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

