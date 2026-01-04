from unittest.mock import patch, MagicMock
from enchanted_surrogates.executors.local_executor import LocalExecutor

@patch("enchanted_surrogates.executors.local_executor.run_simulation_task")
def test_execute_registers_futures(mock_run_simulation_task):
    executor_input = [
        ("test_run_dir_0", {"a": 1, "b": 2}),
        ("test_run_dir_1", {"c": 2})
    ]

    # Fixed return values for the mocked/patched functions
    sampler = MagicMock()
    mock_run_simulation_task.return_value = "FUTURE"

    executor = LocalExecutor(
        sampler_config={"type": "mock"},
        runner_config={"type": "mock"}
    )

    executor.execute(executor_input, sampler)

    # Executor should not directly call get next samples
    assert sampler.get_next_samples.call_count == 0
    assert sampler.register_future.call_count == 2
    assert mock_run_simulation_task.call_count == 2