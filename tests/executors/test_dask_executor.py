import os
import pytest
from unittest.mock import MagicMock, patch

from enchanted_surrogates.executors.dask_executor import DaskExecutor

@patch("enchanted_surrogates.executors.dask_executor.import_sampler")
@patch("enchanted_surrogates.executors.dask_executor.Client")
@patch("enchanted_surrogates.executors.dask_executor.LocalCluster")
def test_start_cluster_uses_LocalCluster(mock_LocalCluster, mock_Client, mock_import_sampler, fake_sampler):
    # start_cluster works with LocalCluster config
    sampler = fake_sampler([])
    mock_import_sampler.return_value = sampler
    mock_cluster = MagicMock()
    mock_LocalCluster.return_value = mock_cluster
    mock_Client.return_value = MagicMock()

    executor = DaskExecutor(
        base_run_dir="/tmp/test_runs",
        sampler_config={"type": "mock"},
        runner_config={"type": "mock"},
        LocalCluster_config={"n_workers": 2}
    )

    executor.start_cluster()

    assert mock_LocalCluster.call_count == 1
    assert executor.expected_number_of_workers == 2
    assert executor.cluster == mock_cluster
    assert executor.client == mock_Client.return_value

@patch("enchanted_surrogates.executors.dask_executor.import_sampler")
@patch("enchanted_surrogates.executors.dask_executor.Client")
@patch("enchanted_surrogates.executors.dask_executor.SLURMCluster")
def test_start_cluster_uses_SLURMCluster(mock_SLURM, mock_Client, mock_import_sampler, tmp_path, fake_sampler):
    # start_cluster works with SLURM Cluster config
    sampler = fake_sampler([])
    mock_import_sampler.return_value = sampler
    mock_cluster = MagicMock()
    mock_SLURM.return_value = mock_cluster
    mock_Client.return_value = MagicMock()

    slurm_config = {"processes": 2}
    executor = DaskExecutor(
        base_run_dir=str(tmp_path),
        sampler_config={"type": "mock"},
        runner_config={"type": "mock"},
        SLURMcluster_config=slurm_config,
        scale_n_jobs=3
    )

    executor.start_cluster()

    assert mock_SLURM.call_count == 1
    assert mock_cluster.scale.call_count == 1
    assert executor.cluster == mock_cluster

    expected_slurm_out_dir = os.path.join(tmp_path, "worker_out_DaskExecutor")
    assert os.path.exists(expected_slurm_out_dir)

    # scale_n_jobs times processes
    assert executor.expected_number_of_workers == 6

@patch("enchanted_surrogates.executors.dask_executor.import_sampler")
@patch("enchanted_surrogates.executors.dask_executor.make_run_dir", return_value="/tmp/run_1")
def test_submit_batch_submits_futures(mock_make_run_dir, mock_import_sampler, fake_sampler):
    # submit_batch creates futures for each sample given
    samples = [
        {"a": 1}, {"b": 2}
    ]
    sampler = fake_sampler(samples)

    mock_import_sampler.return_value = sampler
    mock_future = MagicMock()
    mock_client = MagicMock()
    mock_client.submit.return_value = mock_future

    executor = DaskExecutor(
        base_run_dir="/tmp/test_runs",
        sampler_config={"type": "mock"},
        runner_config={"type": "mock"}
    )

    futures = executor.submit_batch(samples, client=mock_client)

    assert mock_make_run_dir.call_count == len(samples)
    assert mock_client.submit.call_count == len(samples)
    assert len(futures) == 2

@patch("enchanted_surrogates.executors.dask_executor.as_completed")
@patch("enchanted_surrogates.executors.dask_executor.import_sampler")
@patch("enchanted_surrogates.executors.dask_executor.DaskExecutor.submit_batch")
def test_start_runs_creates_futures_from_samples(mock_submit_batch, mock_import_sampler, mock_as_completed, fake_sampler, tmp_path):
    # For each sample dict, a future should be created
    sampler = fake_sampler([
        [{"x": 1}, {"y": 2}], [{"z": 3}]
    ])

    # Mock values for sampler and futures
    mock_import_sampler.return_value = sampler
    mock_future = MagicMock()
    mock_future.result.return_value = {"success": True, "value": 1}
    mock_as_completed.return_value = [mock_future]
    mock_submit_batch.side_effect = [[mock_future, mock_future], [mock_future]]

    base_dir = os.path.join(tmp_path, "runs")
    executor = DaskExecutor(
        base_run_dir=base_dir,
        sampler_config={"type": "mock"},
        runner_config={"type": "mock_runner"},
    )

    # Provide mock client so start_cluster is not called
    executor.client = MagicMock()
    executor.start_runs()

    # Sampler consumed 2 batches
    assert sampler.get_next_samples.call_count == 2
    assert mock_submit_batch.call_count == 2

    assert os.path.exists(os.path.join(base_dir, "ENCHANTED.FINISHED"))
