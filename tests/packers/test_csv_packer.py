import pandas as pd

from enchanted_surrogates.packers.csv_packer import CsvPacker


def _packer(tmp_path):
    return CsvPacker(base_run_dir=str(tmp_path))


def _rows(tmp_path):
    return pd.read_csv(tmp_path / "enchanted_packed.csv")


def test_packing_a_run_dir_twice_gives_one_row(tmp_path):
    packer = _packer(tmp_path)
    packer.pack_run_dir(str(tmp_path / "r0"), {"success": True, "x": 1})
    packer.pack_run_dir(str(tmp_path / "r0"), {"success": True, "x": 1})
    assert len(_rows(tmp_path)) == 1


def test_repacking_overwrites_the_row_in_place(tmp_path):
    packer = _packer(tmp_path)
    for i in range(3):
        packer.pack_run_dir(str(tmp_path / f"r{i}"), {"success": True, "x": i})
    packer.pack_run_dir(str(tmp_path / "r1"), {"success": False, "x": 10})

    rows = _rows(tmp_path)
    assert list(rows.x) == [0, 10, 2]                      # same position, new value
    assert list(rows.success) == [True, False, True]
    assert [r.rsplit("/", 1)[1] for r in rows.run_dir] == ["r0", "r1", "r2"]


def test_repacking_with_a_new_column_keeps_the_other_rows(tmp_path):
    packer = _packer(tmp_path)
    packer.pack_run_dir(str(tmp_path / "r0"), {"x": 0})
    packer.pack_run_dir(str(tmp_path / "r1"), {"x": 1})
    packer.pack_run_dir(str(tmp_path / "r1"), {"x": 5, "y": 7})

    rows = _rows(tmp_path)
    assert len(rows) == 2
    assert list(rows.x) == [0, 5]
    assert pd.isna(rows.y.iloc[0]) and rows.y.iloc[1] == 7


def test_existing_duplicates_collapse_to_one_row_when_repacked(tmp_path):
    packer = _packer(tmp_path)
    run = str(tmp_path / "r0")
    pd.DataFrame([{"run_dir": run, "x": 1}, {"run_dir": run, "x": 2}, {"run_dir": str(tmp_path / "r1"), "x": 3}]).to_csv(
        tmp_path / "enchanted_packed.csv", index=False)
    packer.pack_run_dir(run, {"x": 9})

    rows = _rows(tmp_path)
    assert len(rows) == 2
    assert list(rows.x) == [9, 3]
