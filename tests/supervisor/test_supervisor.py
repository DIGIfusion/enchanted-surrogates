import pandas as pd
from enchanted_surrogates.supervisor.supervisor import Supervisor
from types import SimpleNamespace

def test_all_processes_done_returns_correct_values(tmp_path, patch_supervisor_imports):
    patch_supervisor_imports()
    supervisor = Supervisor(make_args(tmp_path))

    one_run_dir = tmp_path / "0_0"
    one_run_dir.mkdir()
    assert supervisor.all_processes_done() is False

    pd.DataFrame([{"x": 1}]).to_csv(one_run_dir / "enchanted_datapoint.csv", index=False)
    assert supervisor.all_processes_done() is True

def test_create_dataset_combines_csv_files(tmp_path, patch_supervisor_imports):
    patch_supervisor_imports()
    supervisor = Supervisor(make_args(tmp_path))

    for i in range(3):
        d = tmp_path / f"{i}_0"
        d.mkdir()
        pd.DataFrame([{"x": i}]).to_csv(d / "enchanted_datapoint.csv", index=False)

    df = supervisor.create_dataset()
    assert len(df) == 3
    assert set(df["x"]) == {0, 1, 2}

def test_start_calls_execute_for_each_sample(tmp_path, patch_supervisor_imports):
    sampler, executor = patch_supervisor_imports([
        [{"a": 1}, {"b": 2}],
        [{"a": 3}],
    ])

    supervisor = Supervisor(make_args(tmp_path))
    supervisor.start()

    assert sampler.get_next_samples.call_count == 2
    assert executor.execute.call_count == 2

def make_args(tmp_path, summary="csv"):
    """
    Helper function to create constructor arguments
    """
    return SimpleNamespace(
        executor={"type": "mock"},
        sampler={"type": "mock"},
        supervisor={
            "base_run_dir": str(tmp_path),
            "summary_datatype": summary,
        },
    )
