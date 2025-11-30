from unittest.mock import patch, MagicMock
from enchanted_surrogates.executors.joblib_executor import JoblibExecutor

@patch("enchanted_surrogates.executors.joblib_executor.joblib.delayed")
@patch("enchanted_surrogates.executors.joblib_executor.joblib.Parallel")
@patch("enchanted_surrogates.executors.joblib_executor.run_simulation_task")
@patch("enchanted_surrogates.executors.joblib_executor.import_sampler")
def test_start_runs_registers_futures(mock_import_sampler, mock_run_simulation_task, mock_parallel, mock_delayed, fake_sampler):
    sampler = fake_sampler([
        [{"a": 1}, {"b": 2}],
        [{"a": 3}]
    ])

    mock_import_sampler.return_value = sampler
    mock_run_simulation_task.return_value = "FUTURE"

    # Mocking Parallel and Delayed from joblib so all tasks are actually executed right away
    parallel_instance = MagicMock()
    mock_parallel.return_value = parallel_instance
    parallel_instance.side_effect = lambda fn: list(fn)
    mock_delayed.side_effect = lambda fn: lambda *args, **kwargs: fn(*args, **kwargs)

    executor = JoblibExecutor(
        sampler_config={"type": "mock"},
        runner_config={"type": "mock_runner"},
        base_run_dir="/tmp/test_joblib_runs",
    )

    executor.start_runs()

    assert sampler.has_budget == False
    # These are called for every list in samples
    assert sampler.get_next_samples.call_count == 2
    assert sampler.register_futures.call_count == 2
    # This for every dict in samples
    assert mock_run_simulation_task.call_count == 3
