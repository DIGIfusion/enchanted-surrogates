import pytest

from enchanted_surrogates.packers.multi_packer import MultiPacker

CONFIG = [{"type": "ascii_bin_to_hdf5_packer"}, {"type": "csv_packer"}]


def test_on_existing_is_forwarded_to_the_sub_packers(tmp_path):
    multi = MultiPacker(base_run_dir=str(tmp_path), on_existing="overwrite", packers=CONFIG)
    assert [p.on_existing for p in multi.packers] == ["overwrite", "overwrite"]


def test_a_sub_packer_can_set_its_own_on_existing(tmp_path):
    configs = [{"type": "ascii_bin_to_hdf5_packer", "on_existing": "overwrite"}, {"type": "csv_packer"}]
    multi = MultiPacker(base_run_dir=str(tmp_path), packers=configs)
    assert [p.on_existing for p in multi.packers] == ["overwrite", "skip"]


def test_the_default_is_skip_for_every_sub_packer(tmp_path):
    multi = MultiPacker(base_run_dir=str(tmp_path), packers=CONFIG)
    assert [p.on_existing for p in multi.packers] == ["skip", "skip"]
