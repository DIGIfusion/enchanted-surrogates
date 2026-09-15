"""
Tests that verify enchanted_dataset.csv contains the expected, complete data
after a workflow finishes - i.e. every submitted sample produced exactly one
successful row, and no data was lost or duplicated while combining the
per-run enchanted_datapoint.csv files into the summary.
"""

import math

import pytest

from workflow_tests.utils.test_utils import get_run_dir_count, read_summary_file


@pytest.mark.parametrize("config_file", [
    "test_configs/full_workflow_local.yaml",       # simple: one sampler, one runner
    "test_configs/full_workflow_joblib.yaml",       # simple: one sampler, one runner
    "test_configs/full_workflow_dask.yaml",         # simple: one sampler, one runner
    "test_configs/sequential_local.yaml",           # sequential: one sampler, two runners
    "test_configs/nested_sequential.yaml",          # nested: two samplers, each with two sequential runners
])
def test_dataset_contains_all_completed_runs(tmp_path, run_config, config_file):
    supervisor = run_config(config_file)

    # The dataset only contains rows for the deepest (final) run group, so
    # the expected row count is the product of every group's sampler budget.
    expected_rows = math.prod(
        group.sampler.budget for group in supervisor.nested_groups
    )

    # Each runner used anywhere in the workflow writes its own
    # "success_<runner_name>" column onto every row it touches.
    expected_success_columns = {
        f"success_{runner['__runner_name']}"
        for group in supervisor.nested_groups
        for runner in group.runners
    }

    rows = read_summary_file(tmp_path)

    assert len(rows) == expected_rows

    # Each of a group's sequential runners gets its own run directory, so a
    # group at depth N creates (product of budgets up to N) * (its runner count) dirs.
    expected_run_dirs = sum(
        math.prod(g.sampler.budget for g in supervisor.nested_groups[: depth + 1])
        * len(group.runners)
        for depth, group in enumerate(supervisor.nested_groups)
    )
    assert get_run_dir_count(tmp_path / "data") == expected_run_dirs

    # No run should be missing or duplicated
    run_dirs = [row["run_dir"] for row in rows]
    assert len(set(run_dirs)) == expected_rows

    # Every run should have completed successfully, with no runner failing
    # and no nan values in the primary numeric output
    for row in rows:
        assert row["success"] is True
        assert not (isinstance(row["output"], float) and math.isnan(row["output"]))
        for success_column in expected_success_columns:
            assert row[success_column] is True
