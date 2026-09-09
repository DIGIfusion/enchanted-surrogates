"""
Tests that verify enchanted_dataset.csv contains the expected, complete data
after a workflow finishes - i.e. every submitted sample produced exactly one
successful row, and no data was lost or duplicated while combining the
per-run enchanted_datapoint.csv files into the summary.
"""

import pytest
from workflow_tests.utils.test_utils import *


@pytest.mark.parametrize("config_file", [
    "test_configs/full_workflow_local.yaml",
    "test_configs/full_workflow_joblib.yaml",
    "test_configs/full_workflow_dask.yaml"
])
def test_dataset_contains_all_completed_runs(tmp_path, run_config, config_file):
    supervisor = run_config(config_file)
    run_group = supervisor.nested_groups[0]
    budget = run_group.sampler.budget

    rows = read_summary_file(tmp_path)

    # Every submitted sample should have produced exactly one row
    assert len(rows) == budget
    assert get_run_dir_count(tmp_path / "data") == budget

    # No run should be missing or duplicated
    run_dirs = [row["run_dir"] for row in rows]
    assert len(set(run_dirs)) == budget

    # Every run should have completed successfully with its parameters recorded
    for row in rows:
        assert row["success_r1"] is True
        assert row["c1"] is not None
        assert row["c2"] is not None
