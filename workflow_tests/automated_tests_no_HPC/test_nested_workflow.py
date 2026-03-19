"""
Basic tests for the full workflow when using nested executors and samplers.
"""

from enchanted_surrogates.executors import LocalExecutor, JoblibExecutor, DaskExecutor
from enchanted_surrogates.supervisor.supervisor import Supervisor
from workflow_tests.utils.test_utils import *


def test_nested_dask_executors(tmp_path, run_config):
    supervisor: Supervisor = run_config("test_configs/nested_dask_executors.yaml")
    assert len(supervisor.groups) == 2

    budget_first = supervisor.groups[0].sampler.budget
    budget_second = supervisor.groups[1].sampler.budget
    assert (
        get_run_dir_count(tmp_path / "data")
        == budget_first + budget_first * budget_second
    )


def test_nested_dask_executors_with_executor_reuse(tmp_path, run_config):
    supervisor = run_config("test_configs/nested_executor_reuse.yaml")
    assert len(supervisor.groups) == 3

    assert supervisor.groups[0].executors[0] == supervisor.groups[1].executors[0]
    assert supervisor.groups[1].executors[0] == supervisor.groups[2].executors[0]

    budget_first = supervisor.groups[0].sampler.budget
    budget_second = supervisor.groups[1].sampler.budget
    budget_third = supervisor.groups[2].sampler.budget
    assert get_run_dir_count(tmp_path / "data") == (
        budget_first
        + budget_first * budget_second
        + budget_first * budget_second * budget_third
    )
