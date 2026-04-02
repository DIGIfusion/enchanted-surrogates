"""
Workflow tests for verifying a success column is shown for each runner in a nested
or sequential workflow.
"""

from enchanted_surrogates.supervisor.supervisor import Supervisor, RunGroup
from workflow_tests.utils.test_utils import *

def get_all_runner_names(groups: list[RunGroup]) -> list[str]:
    all_runners = []
    for group in groups:
        for runner in group.runners:
            all_runners.append(list(runner.keys())[0])
    return all_runners

def test_runner_failures_are_seen_in_summary(tmp_path, run_config):
    """
    Nested config that creates 4+(4*3) samples.
    Runner fails if the sum of samples is 2.
    So (1*3)+4 failures are expected to be seen.
    """
    total_sample_count = 4*3
    failures = [1*3, 4]

    supervisor: Supervisor = run_config("test_configs/runner_failure_nested.yaml")

    all_runners = get_all_runner_names(supervisor.nested_groups)
    assert len(all_runners) == len(failures)

    # TODO assert total rows is 12
    # TODO assert failures on first runner is 1 * 3
    # TODO assert failures on second (final) runner is 4
    # TODO assert all others should say success=True
    summary = read_summary_file(tmp_path)
    assert len(summary) == total_sample_count

    for i, runner in enumerate(all_runners):
        # last one is just named success with to suffix
        name: str = f"success_{runner}" if i < len(all_runners) - 1 else "success"
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
