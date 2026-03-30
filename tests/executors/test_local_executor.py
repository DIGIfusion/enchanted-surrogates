from unittest.mock import patch, MagicMock
from enchanted_surrogates.executors.local_executor import LocalExecutor


@patch("enchanted_surrogates.executors.local_executor.run_simulation_task")
def test_execute(mock_run_simulation_task):
    executor_input = [
        ("test_run_dir_0", {"a": 1, "b": 2}),
        ("test_run_dir_1", {"c": 2}),
    ]

    # Fixed return values for the mocked/patched functions
    mock_run_simulation_task.return_value = "EXECUTE"

    executor = LocalExecutor()

    executor.execute(executor_input, runner_config={"type": "mock"})

    # Executor should not directly call get next samples
    assert mock_run_simulation_task.call_count == 2
