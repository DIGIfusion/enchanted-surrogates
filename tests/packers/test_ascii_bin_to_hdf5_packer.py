import h5py
import numpy as np

from enchanted_surrogates.packers.ascii_bin_to_hdf5_packer import AsciiBinToHdf5Packer


def _make_run_dir(path, files):
    path.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (path / name).write_bytes(content) if isinstance(content, bytes) else (path / name).write_text(content)
    return str(path)


def _packer(tmp_path):
    return AsciiBinToHdf5Packer(base_run_dir=str(tmp_path), hdf5_path=str(tmp_path / "packed.h5"))


def test_packing_a_run_dir_twice_gives_the_same_state_as_once(tmp_path):
    packer = _packer(tmp_path)
    run = _make_run_dir(tmp_path / "data" / "r0", {"a.txt": "hello", "b.bin": bytes([0, 1, 2, 255])})
    packer.pack_run_dir(run)
    packer.pack_run_dir(run)

    with h5py.File(tmp_path / "packed.h5", "r") as f:
        assert list(f["runs"]) == ["r0"]
        assert f["runs/r0/a.txt"][()].decode() == "hello"
        assert np.array_equal(f["runs/r0/b.bin"][()], [0, 1, 2, 255])


def test_repacking_replaces_the_packed_content_with_the_current_content(tmp_path):
    packer = _packer(tmp_path)
    run = _make_run_dir(tmp_path / "data" / "r0", {"a.txt": "first", "gone.txt": "x"})
    packer.pack_run_dir(run)

    (tmp_path / "data" / "r0" / "gone.txt").unlink()          # e.g. cleaned up by the supervisor
    (tmp_path / "data" / "r0" / "a.txt").write_text("second")  # or rerun
    packer.pack_run_dir(run)

    with h5py.File(tmp_path / "packed.h5", "r") as f:
        assert f["runs/r0/a.txt"][()].decode() == "second"
        assert "gone.txt" not in f["runs/r0"]


def test_repacking_one_run_leaves_the_others_alone(tmp_path):
    packer = _packer(tmp_path)
    r0 = _make_run_dir(tmp_path / "data" / "r0", {"a.txt": "zero"})
    r1 = _make_run_dir(tmp_path / "data" / "r1", {"a.txt": "one"})
    packer.pack_run_dir(r0)
    packer.pack_run_dir(r1)
    packer.pack_run_dir(r0)

    with h5py.File(tmp_path / "packed.h5", "r") as f:
        assert sorted(f["runs"]) == ["r0", "r1"]
        assert f["runs/r1/a.txt"][()].decode() == "one"
