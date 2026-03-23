"""
Basic tests for the sequential workflow.
"""

import pytest
from workflow_tests.utils.test_utils import *
from enchanted_surrogates.runners.base_runner import Runner

class EvalRunner(Runner):
    def __init__(self, clause: str, **kwargs):
        self.clause = clause

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        # TODO unpack params so they can be used in eval
        result = eval(self.clause)
        # TODO write result


def test_sequential_workflow(tmp_path, run_config):
    pass
    return

    supervisor = run_config("test_configs/sequential_local.yaml")
    run_group = supervisor.nested_groups[0]

    # This should create {budget} folders
    assert get_run_dir_count(tmp_path / "data") == run_group.sampler.budget

    # TODO: read summary CSV and compare to hardcoded expected csv
    summary = read_summary_file(tmp_path)
