"""
Basic tests for the sequential workflow.
"""

from pathlib import Path

import pytest
from workflow_tests.utils.test_utils import get_run_dir_count, read_summary_file
from enchanted_surrogates.packers.ascii_bin_to_hdf5_packer import AsciiBinToHdf5Packer
from enchanted_surrogates.supervisor.supervisor import Supervisor

def test_sequential_workflow(tmp_path, run_config):
    supervisor: Supervisor = run_config("test_configs/sequential_local.yaml")
    sequence_count = len(supervisor.args.supervisor["run_order"][0]["runner"])
    
    run_group = supervisor.nested_groups[0]

    # Sequence count in config matches one in Supervisor
    assert len(run_group.runners) == sequence_count

    # This should create {budget * sequence_count} folders
    assert get_run_dir_count(tmp_path / "data") == run_group.sampler.budget * sequence_count

    # Summary file should only contain output from the final sequence
    summary = read_summary_file(tmp_path)
    assert len(summary) == run_group.sampler.budget

    for row in summary:
        assert row["output"] == pytest.approx(12.0)

def test_nested_sequential_workflow(tmp_path, run_config):
    supervisor: Supervisor = run_config("test_configs/nested_sequential.yaml")
    sequence_count_1 = len(supervisor.args.supervisor["run_order"][0]["runner"])
    sequence_count_2 = len(supervisor.args.supervisor["run_order"][1]["runner"])

    run_group_1 = supervisor.nested_groups[0]
    run_group_2 = supervisor.nested_groups[1]
    budget_1 = run_group_1.sampler.budget
    budget_2 = run_group_2.sampler.budget

    # Sequence count in config matches one in Supervisor
    assert len(run_group_1.runners) == sequence_count_1
    assert len(run_group_2.runners) == sequence_count_2

    # This should create {bud1 * seq1} + {bud1} * {bud2 * seq2} folders
    assert get_run_dir_count(tmp_path / "data") == (
        budget_1 * sequence_count_1 + budget_1 * budget_2 * sequence_count_2
    )

    # Summary file should only contain output from the final sequence and final nesting level
    summary = read_summary_file(tmp_path)
    assert len(summary) == budget_1 * budget_2

    # All results after first nested run will be 12
    # All results after second nested run will be 12+10
    for row in summary:
        assert row["output"] == pytest.approx(22.0)


def test_nested_sequential_workflow_with_packer(tmp_path, run_config):
    supervisor = run_config("test_configs/nested_sequential_with_packer.yaml", call_start=False)

    packed_hdf5 = tmp_path / "packed_runs.h5"
    for group in supervisor.nested_groups:
        if group.packers is not None:
            for packer in group.packers:
                packer.hdf5_path = str(packed_hdf5)

    supervisor.start()

    assert len(supervisor.nested_groups) == 2
    assert get_run_dir_count(tmp_path / "data") > 0
    assert len(read_summary_file(tmp_path)) > 0
    assert packed_hdf5.exists()

    first_run_dir = next((path for path in (tmp_path / "data").iterdir() if path.is_dir()), None)
    assert first_run_dir is not None

    packer = AsciiBinToHdf5Packer(hdf5_path=str(packed_hdf5))
    unpacked_dir = packer.unpack_run_dir(str(first_run_dir), str(tmp_path / "unpacked"))

    assert Path(unpacked_dir).exists()
    assert any(Path(unpacked_dir).iterdir())
