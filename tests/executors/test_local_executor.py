from unittest.mock import patch
from enchanted_surrogates.executors.local_executor import LocalExecutor

@patch("enchanted_surrogates.executors.local_executor.run_simulation_task")
@patch("enchanted_surrogates.executors.local_executor.import_sampler")
def test_start_runs_registers_futures(mock_import_sampler, mock_run_simulation_task, fake_sampler):
    sampler = fake_sampler([
        [{"a": 1}, {"b": 2}],
        [{"a": 3}]
    ])

    # Fixed return values for the mocked/patched functions
    mock_import_sampler.return_value = sampler
    mock_run_simulation_task.return_value = "FUTURE"

    executor = LocalExecutor(
        sampler_config={"type": "mock"},
        runner_config={"type": "mock"},
        base_run_dir="/tmp/test_runs"
    )

    executor.start_runs()

    # 2 batches of samples and total 3 samples should result in 3 simulation tasks
    assert sampler.has_budget == False
    assert sampler.get_next_samples.call_count == 2
    assert sampler.register_future.call_count == 3
    assert mock_run_simulation_task.call_count == 3