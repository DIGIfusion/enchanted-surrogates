"""
Workflow tests for verifying a success column is shown for each runner in a nested
or sequential workflow.
"""

import pytest
from enchanted_surrogates.supervisor.supervisor import Supervisor, RunGroup
from workflow_tests.utils.test_utils import *

def get_all_runner_names(groups: list[RunGroup]) -> list[str]:
    all_runners = []
    for group in groups:
        for runner in group.runners:
            all_runners.append(runner["__runner_name"])
    return all_runners

@pytest.mark.parametrize("data", [
    {"config": "test_configs/runner_failure_nested.yaml", "sample_count": 4*3, "failures": [1*3, 4]},
    {"config": "test_configs/runner_failure_sequential.yaml", "sample_count": 4, "failures": [1, 2]},
    {"config": "test_configs/runner_failure_simple.yaml", "sample_count": 4, "failures": [1]}
])
def test_runner_failures_are_seen_in_summary(tmp_path, run_config, data):
    """
    - Asserts that the amount of rows in final summary file is *sample_count*
    - Asserts all rows have a boolean value for 'success'
    - *failures[n]* are expected to be seen for the nth runner in order of execution.
    """
    config_file = data["config"]
    total_sample_count = data["sample_count"]
    failures = data["failures"]

    supervisor: Supervisor = run_config(config_file)

    all_runners = get_all_runner_names(supervisor.nested_groups)
    assert len(all_runners) == len(failures)

    summary = read_summary_file(tmp_path)
    assert len(summary) == total_sample_count

    # Verify amount of failures for each runner
    for i, runner in enumerate(all_runners):
        name: str = f"success_{runner}"
        success_count = 0
        failure_count = 0

        for row in summary:
            assert isinstance(row[name], bool)
            if row[name]:
                success_count += 1
            else:
                failure_count += 1

        assert failure_count == failures[i]
        assert success_count + failure_count == total_sample_count

    # Verify 'success' matches that of the final runner
    final_runner = all_runners[-1]
    for row in summary:
        assert row["success"] == row[f"success_{final_runner}"]
