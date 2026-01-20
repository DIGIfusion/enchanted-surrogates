import os
from unittest.mock import MagicMock, patch
from enchanted_surrogates.executors.dask_executor import DaskExecutor

@patch("enchanted_surrogates.executors.dask_executor.Client")
@patch("enchanted_surrogates.executors.dask_executor.LocalCluster")
def test_start_cluster_uses_LocalCluster(mock_LocalCluster, mock_Client, tmp_path):
    # start_cluster works with LocalCluster config
    mock_cluster = MagicMock()
    mock_LocalCluster.return_value = mock_cluster
    mock_Client.return_value = MagicMock()

    executor = DaskExecutor(
        runner_config={"type": "mock"},
        LocalCluster_config={"n_workers": 2}
    )
    executor.base_run_dir = tmp_path
    executor.start_cluster()

    assert mock_LocalCluster.call_count == 1
    assert executor.expected_number_of_workers == 2
    assert executor.cluster == mock_cluster
    assert executor.client == mock_Client.return_value

@patch("enchanted_surrogates.executors.dask_executor.Client")
@patch("enchanted_surrogates.executors.dask_executor.SLURMCluster")
def test_start_cluster_uses_SLURMCluster(mock_SLURM, mock_Client, tmp_path):
    # start_cluster works with SLURM Cluster config
    mock_cluster = MagicMock()
    mock_SLURM.return_value = mock_cluster
    mock_Client.return_value = MagicMock()

    slurm_config = {"processes": 2}
    executor = DaskExecutor(
        runner_config={"type": "mock"},
        SLURMcluster_config=slurm_config,
        scale_n_jobs=3
    )
    executor.base_run_dir = tmp_path
    executor.start_cluster()

    assert mock_SLURM.call_count == 1
    assert mock_cluster.scale.call_count == 1
    assert executor.cluster == mock_cluster

    expected_slurm_out_dir = os.path.join(tmp_path, "worker_out_DaskExecutor")
    assert os.path.exists(expected_slurm_out_dir)

    # scale_n_jobs times processes
    assert executor.expected_number_of_workers == 6

def test_submit_batch_submits_futures():
    executor_input = [
        ("test_run_dir_0", {"a": 1, "b": 2}),
        ("test_run_dir_1", {"c": 2})
    ]

    mock_future = MagicMock()
    mock_client = MagicMock()
    mock_client.submit.return_value = mock_future

    executor = DaskExecutor(
        runner_config={"type": "mock"}
    )

    futures = executor.submit_batch(executor_input, client=mock_client)

    assert mock_client.submit.call_count == 2
    assert len(futures) == 2

@patch("enchanted_surrogates.executors.dask_executor.as_completed")
@patch("enchanted_surrogates.executors.dask_executor.DaskExecutor.submit_batch")
def test_execute_creates_futures_from_samples(mock_submit_batch, mock_as_completed):
    executor_input = [
        ("test_run_dir_0", {"a": 1, "b": 2}),
        ("test_run_dir_1", {"c": 2})
    ]

    # Mock values for sampler and futures
    sampler = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = {"success": True, "value": 1}
    mock_as_completed.return_value = [mock_future]
    mock_submit_batch.side_effect = ([[mock_future, mock_future], [mock_future]], [])

    executor = DaskExecutor(
        runner_config={"type": "mock_runner"}
    )

    # Provide mock client so start_cluster is not called
    executor.client = MagicMock()
    executor.execute(executor_input, sampler)

    # Sampler consumed 2 batches
    assert sampler.get_next_samples.call_count == 0
    assert mock_submit_batch.call_count == 1
