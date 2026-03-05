"""
Basic tests for the full workflow with different executors.
Each executor should be tested in a separate function to avoid conflicts."""

import pytest
from workflow_tests.utils.test_utils import *

def test_full_workflow_local(tmp_path, run_config):
    config = "test_configs/full_workflow_local.yaml"
    supervisor = run_config(config)
    run_group = supervisor.groups[0]

    # This should create {budget} folders
    assert get_run_dir_count(tmp_path) == run_group.sampler.budget


def test_full_workflow_joblib(tmp_path, run_config):
    config = "test_configs/full_workflow_joblib.yaml"
    supervisor = run_config(config)
    run_group = supervisor.groups[0]

    # This should create {budget} folders
    assert get_run_dir_count(tmp_path) == run_group.sampler.budget


def test_full_workflow_dask(tmp_path, run_config):
    config = "test_configs/full_workflow_dask.yaml"
    supervisor = run_config(config)
    run_group = supervisor.groups[0]

    # This should create {budget} folders
    assert get_run_dir_count(tmp_path) == run_group.sampler.budget
