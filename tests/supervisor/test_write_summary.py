"""write_summary keeps every value under its own column when the columns of the batches differ."""
from types import SimpleNamespace

import pandas as pd

from enchanted_surrogates.supervisor.supervisor import Supervisor, append_aligned_csv


def write(tmp_path, batches):
    sup = SimpleNamespace(base_run_dir=str(tmp_path))
    for i, df in enumerate(batches):
        Supervisor.write_summary(sup, df, write_mode="w" if i == 0 else "a")
    return pd.read_csv(tmp_path / "enchanted_dataset.csv")


def test_same_columns_in_another_order_are_aligned(tmp_path):
    out = write(tmp_path, [pd.DataFrame({"a": [1, 2], "b": [3, 4]}), pd.DataFrame({"b": [30], "a": [10]})])
    assert out.to_dict("list") == {"a": [1, 2, 10], "b": [3, 4, 30]}


def test_a_new_column_in_a_later_batch_is_added_not_misaligned(tmp_path):
    # the failure of the v3 production run: a key that only some runs return (volume_scaling_error) appeared in batch 2
    b0 = pd.DataFrame({"success": [True, False], "q95": [3.0, None], "run_dir": ["d0", "d1"]})
    b1 = pd.DataFrame({"success": [False], "volume_scaling_error": ["pass 1: failed"], "q95": [None], "run_dir": ["d2"]})
    b2 = pd.DataFrame({"success": [True], "q95": [4.0], "run_dir": ["d3"]})
    out = write(tmp_path, [b0, b1, b2])
    assert list(out.columns) == ["success", "q95", "run_dir", "volume_scaling_error"]
    assert out.run_dir.tolist() == ["d0", "d1", "d2", "d3"]
    assert out.q95.tolist()[0] == 3.0 and out.q95.tolist()[3] == 4.0
    assert out.volume_scaling_error.isna().tolist() == [True, True, False, True]


def test_a_missing_column_is_left_empty(tmp_path):
    out = write(tmp_path, [pd.DataFrame({"a": [1], "b": [2], "c": [3]}), pd.DataFrame({"c": [30], "a": [10]})])
    assert out.loc[1].isna().tolist() == [False, True, False] and out.c.tolist() == [3, 30]


def test_append_to_a_missing_file_writes_a_header(tmp_path):
    path = tmp_path / "x.csv"
    append_aligned_csv(str(path), pd.DataFrame({"a": [1], "b": [2]}))
    append_aligned_csv(str(path), pd.DataFrame({"b": [4], "a": [3]}))
    assert pd.read_csv(path).to_dict("list") == {"a": [1, 3], "b": [2, 4]}


def test_no_temporary_file_is_left_behind(tmp_path):
    write(tmp_path, [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2], "z": [3]})])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["enchanted_dataset.csv"]
