from unittest.mock import patch, MagicMock
from enchanted_surrogates.executors.joblib_executor import JoblibExecutor

@patch("enchanted_surrogates.executors.joblib_executor.joblib.delayed")
@patch("enchanted_surrogates.executors.joblib_executor.joblib.Parallel")
@patch("enchanted_surrogates.executors.joblib_executor.run_simulation_task")
def test_execute_registers_futures(mock_run_simulation_task, mock_parallel, mock_delayed):
    executor_input = [
        ("test_run_dir_0", {"a": 1, "b": 2}),
        ("test_run_dir_1", {"c": 2})
    ]

    sampler = MagicMock()
    mock_run_simulation_task.return_value = "FUTURE"

    # Mocking Parallel and Delayed from joblib so all tasks are actually executed right away
    parallel_instance = MagicMock()
    mock_parallel.return_value = parallel_instance
    parallel_instance.side_effect = lambda fn: list(fn)
    mock_delayed.side_effect = lambda fn: lambda *args, **kwargs: fn(*args, **kwargs)

    executor = JoblibExecutor(
        runner_config={"type": "mock_runner"}
    )

    executor.execute(executor_input, sampler)

    # Executor should not directly call get next samples
    assert sampler.get_next_samples.call_count == 0
    assert sampler.register_futures.call_count == 1
    assert mock_run_simulation_task.call_count == 2
