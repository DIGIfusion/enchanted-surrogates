"""
Basic tests for the full workflow with different executors.
Each executor should be tested in a separate function to avoid conflicts."""

import pytest
from workflow_tests.utils.test_utils import *

@pytest.mark.parametrize("config_file", [
    "test_configs/full_workflow_local.yaml",
    "test_configs/full_workflow_joblib.yaml",
    "test_configs/full_workflow_dask.yaml"
])
def test_full_workflow(tmp_path, run_config, config_file):
    supervisor = run_config(config_file)
    run_group = supervisor.groups[0]

    # This should create {budget} folders
    assert get_run_dir_count(tmp_path / "data") == run_group.sampler.budget
