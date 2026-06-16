from unittest.mock import patch, MagicMock
from enchanted_surrogates.executors.joblib_executor import JoblibExecutor


@patch("enchanted_surrogates.executors.joblib_executor.joblib.delayed")
@patch("enchanted_surrogates.executors.joblib_executor.joblib.Parallel")
@patch("enchanted_surrogates.executors.joblib_executor.run_simulation_task")
def test_execute(mock_run_simulation_task, mock_parallel, mock_delayed):
    executor_input = [
        ("test_run_dir_0", {"a": 1, "b": 2}),
        ("test_run_dir_1", {"c": 2}),
    ]

    mock_run_simulation_task.return_value = "EXECUTE"

    # Mocking Parallel and Delayed from joblib so all tasks are actually executed right away
    parallel_instance = MagicMock()
    mock_parallel.return_value = parallel_instance
    parallel_instance.side_effect = lambda fn: list(fn)
    mock_delayed.side_effect = lambda fn: lambda *args, **kwargs: fn(*args, **kwargs)

    executor = JoblibExecutor()

    executor.execute(executor_input, runner_config={"type": "mock_runner"})

    # Executor should not directly call get next samples
    assert mock_run_simulation_task.call_count == 2
