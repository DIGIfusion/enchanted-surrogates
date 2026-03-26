"""
Basic tests for the sequential workflow.
"""

import pytest
from workflow_tests.utils.test_utils import *
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
