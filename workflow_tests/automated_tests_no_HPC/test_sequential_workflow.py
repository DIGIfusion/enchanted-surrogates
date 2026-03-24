"""
Basic tests for the sequential workflow.
"""

import pytest
from workflow_tests.utils.test_utils import *
from enchanted_surrogates.supervisor.supervisor import Supervisor

def test_sequential_workflow(tmp_path, run_config):
    supervisor: Supervisor = run_config("test_configs/sequential_local.yaml")
    run_group = supervisor.nested_groups[0]
    sequences = len(run_group.runners)

    # This should create {budget * sequence_count} folders
    assert get_run_dir_count(tmp_path / "data") == run_group.sampler.budget * sequences

    # Summary file should only contain output from the final sequence
    summary = read_summary_file(tmp_path)
    assert len(summary) == run_group.sampler.budget

    for row in summary:
        assert row["output"] == pytest.approx(12.0)
