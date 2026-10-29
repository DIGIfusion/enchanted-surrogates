import pandas as pd

from enchanted_surrogates.packers.csv_packer import CsvPacker


def _packer(tmp_path, **kwargs):
    return CsvPacker(base_run_dir=str(tmp_path), **kwargs)


def _rows(tmp_path):
    return pd.read_csv(tmp_path / "enchanted_packed.csv")


def test_packing_a_run_dir_twice_gives_one_row(tmp_path):
    packer = _packer(tmp_path)
    packer.pack_run_dir(str(tmp_path / "r0"), {"success": True, "x": 1})
    packer.pack_run_dir(str(tmp_path / "r0"), {"success": True, "x": 1})
    assert len(_rows(tmp_path)) == 1


def test_overwrite_replaces_the_row_in_place(tmp_path):
    packer = _packer(tmp_path, on_existing="overwrite")
    for i in range(3):
        packer.pack_run_dir(str(tmp_path / f"r{i}"), {"success": True, "x": i})
    packer.pack_run_dir(str(tmp_path / "r1"), {"success": False, "x": 10})

    rows = _rows(tmp_path)
    assert list(rows.x) == [0, 10, 2]                      # same position, new value
    assert list(rows.success) == [True, False, True]
    assert [r.rsplit("/", 1)[1] for r in rows.run_dir] == ["r0", "r1", "r2"]


def test_overwrite_with_a_new_column_keeps_the_other_rows(tmp_path):
    packer = _packer(tmp_path, on_existing="overwrite")
    packer.pack_run_dir(str(tmp_path / "r0"), {"x": 0})
    packer.pack_run_dir(str(tmp_path / "r1"), {"x": 1})
    packer.pack_run_dir(str(tmp_path / "r1"), {"x": 5, "y": 7})

    rows = _rows(tmp_path)
    assert len(rows) == 2
    assert list(rows.x) == [0, 5]
    assert pd.isna(rows.y.iloc[0]) and rows.y.iloc[1] == 7


def test_overwrite_collapses_existing_duplicates_to_one_row(tmp_path):
    packer = _packer(tmp_path, on_existing="overwrite")
    run = str(tmp_path / "r0")
    pd.DataFrame([{"run_dir": run, "x": 1}, {"run_dir": run, "x": 2}, {"run_dir": str(tmp_path / "r1"), "x": 3}]).to_csv(
        tmp_path / "enchanted_packed.csv", index=False)
    packer.pack_run_dir(run, {"x": 9})

    rows = _rows(tmp_path)
    assert len(rows) == 2
    assert list(rows.x) == [9, 3]


def test_skip_is_the_default_and_keeps_the_existing_row(tmp_path):
    packer = _packer(tmp_path)
    assert packer.on_existing == "skip"
    packer.pack_run_dir(str(tmp_path / "r0"), {"success": True, "x": 1})
    packer.pack_run_dir(str(tmp_path / "r1"), {"success": True, "x": 2})
    packer.pack_run_dir(str(tmp_path / "r0"), {"success": False, "x": 99})

    rows = _rows(tmp_path)
    assert list(rows.x) == [1, 2]
    assert list(rows.success) == [True, True]


def test_new_run_dir_is_appended_without_rereading_the_file(tmp_path, monkeypatch):
    packer = _packer(tmp_path)
    packer.pack_run_dir(str(tmp_path / "r0"), {"success": True, "x": 1})
    packer.pack_run_dir(str(tmp_path / "r1"), {"success": True, "x": 2})   # load cache, then append

    def fail(*args, **kwargs):
        raise AssertionError("csv was reread or rewritten")
    monkeypatch.setattr(pd, "read_csv", fail)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    packer.pack_run_dir(str(tmp_path / "r2"), {"success": False, "x": 3})
    packer.pack_run_dir(str(tmp_path / "r0"), {"success": False, "x": 99})  # skip: no io either
    monkeypatch.undo()

    rows = _rows(tmp_path)
    assert list(rows.x) == [1, 2, 3]
    assert list(rows.success) == [True, True, False]


def test_append_writes_missing_keys_blank_and_numpy_values_like_pandas(tmp_path):
    import numpy as np
    packer = _packer(tmp_path)
    packer.pack_run_dir(str(tmp_path / "r0"), {"a": 1.5, "b": "t", "c": True})
    packer.pack_run_dir(str(tmp_path / "r1"), {"a": np.float64(0.1), "c": np.bool_(False)})
    packer.pack_run_dir(str(tmp_path / "r2"), {"a": float("nan"), "b": "has,comma", "c": np.int64(3)})

    rows = _rows(tmp_path)
    assert list(rows.columns) == ["run_dir", "a", "b", "c"]
    assert rows.a.iloc[1] == 0.1 and pd.isna(rows.a.iloc[2])
    assert pd.isna(rows.b.iloc[1]) and rows.b.iloc[2] == "has,comma"
    assert "np." not in (tmp_path / "enchanted_packed.csv").read_text()


def test_new_column_in_the_middle_of_a_run_keeps_appending_afterwards(tmp_path):
    packer = _packer(tmp_path)
    packer.pack_run_dir(str(tmp_path / "r0"), {"x": 0})
    packer.pack_run_dir(str(tmp_path / "r1"), {"x": 1, "y": 7})   # rewrite with a new column
    packer.pack_run_dir(str(tmp_path / "r2"), {"x": 2})            # append, y blank
    packer.pack_run_dir(str(tmp_path / "r1"), {"x": 5})            # skip

    rows = _rows(tmp_path)
    assert list(rows.x) == [0, 1, 2]
    assert pd.isna(rows.y.iloc[0]) and rows.y.iloc[1] == 7 and pd.isna(rows.y.iloc[2])


def test_resume_with_a_new_packer_skips_packed_runs_and_appends_new_ones(tmp_path):
    first = _packer(tmp_path)
    for i in range(3):
        first.pack_run_dir(str(tmp_path / f"r{i}"), {"success": True, "x": i})

    resumed = _packer(tmp_path)    # new process after a crash: cache is rebuilt from the file
    for i in range(5):
        resumed.pack_run_dir(str(tmp_path / f"r{i}"), {"success": True, "x": 10 + i})

    rows = _rows(tmp_path)
    assert list(rows.x) == [0, 1, 2, 13, 14]


def test_torn_last_line_is_dropped_and_the_run_is_packed_again(tmp_path):
    packer = _packer(tmp_path)
    packer.pack_run_dir(str(tmp_path / "r0"), {"success": True, "x": 0})
    packer.pack_run_dir(str(tmp_path / "r1"), {"success": True, "x": 1})
    path = tmp_path / "enchanted_packed.csv"
    path.write_text(path.read_text() + f"{tmp_path}/r2,Tr")          # killed mid-append

    resumed = _packer(tmp_path)
    resumed.pack_run_dir(str(tmp_path / "r2"), {"success": True, "x": 2})
    resumed.pack_run_dir(str(tmp_path / "r3"), {"success": True, "x": 3})

    rows = _rows(tmp_path)
    assert list(rows.x) == [0, 1, 2, 3]
    assert [r.rsplit("/", 1)[1] for r in rows.run_dir] == ["r0", "r1", "r2", "r3"]


def test_rewrite_leaves_no_temporary_file(tmp_path):
    packer = _packer(tmp_path, on_existing="overwrite")
    packer.pack_run_dir(str(tmp_path / "r0"), {"x": 0})
    packer.pack_run_dir(str(tmp_path / "r0"), {"x": 1})
    assert not (tmp_path / "enchanted_packed.csv.tmp").exists()
