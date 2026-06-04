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
        SLURMcluster_config=slurm_config, scale_n_jobs=3
    )
    executor.base_run_dir = tmp_path
    executor.start_cluster()

    assert mock_SLURM.call_count == 1
    # Implementation may call scale more than once (internal checks), ensure it was called
    assert mock_cluster.scale.call_count >= 1
    assert executor.cluster == mock_cluster

    # scale_n_jobs times processes
    assert executor.expected_number_of_workers == 6


@patch("enchanted_surrogates.executors.dask_executor.Client")
@patch("enchanted_surrogates.executors.dask_executor.SLURMCluster")
def test_start_cluster_uses_SLURMCluster_adapt(
    mock_SLURM, mock_Client, tmp_path
):
    # start_cluster should call adapt when min/max are provided
    mock_cluster = MagicMock()
    mock_SLURM.return_value = mock_cluster
    mock_Client.return_value = MagicMock()

    # processes=2 so expected workers should be min * processes = 3*2 = 6
    num_processes = 2
    slurm_config = {"processes": num_processes}
    executor = DaskExecutor(
        SLURMcluster_config=slurm_config, scale_n_jobs_min=3, scale_n_jobs_max=5
    )
    executor.base_run_dir = tmp_path
    executor.start_cluster()

    num_min_workers = 3*num_processes
    num_max_workers = 5*num_processes

    # SLURMCluster constructed and adapt called with provided bounds
    assert mock_SLURM.call_count == 1
    mock_cluster.adapt.assert_called_once_with(minimum=num_min_workers, maximum=num_max_workers)
    # expected_number_of_workers should be set to scale_n_jobs_min * processes
    assert executor.expected_number_of_workers == 6
    assert executor.cluster == mock_cluster


@patch("enchanted_surrogates.executors.dask_executor.Client")
@patch("enchanted_surrogates.executors.dask_executor.SLURMCluster")
def test_start_cluster_adapt_called_with_bounds(mock_SLURM, mock_Client, tmp_path):
    # Ensure adapt is called when min and max are provided (explicit test)
    mock_cluster = MagicMock()
    mock_SLURM.return_value = mock_cluster
    mock_Client.return_value = MagicMock()

    num_processes = 2
    slurm_config = {"processes": num_processes}
    executor = DaskExecutor(
        SLURMcluster_config=slurm_config, scale_n_jobs_min=4, scale_n_jobs_max=8
    )
    executor.base_run_dir = tmp_path
    executor.start_cluster()

    num_min_workers = 4*num_processes
    num_max_workers = 8*num_processes
    mock_cluster.adapt.assert_called_once_with(minimum=num_min_workers, maximum=num_max_workers)
    # expected_number_of_workers should be min * processes
    assert executor.expected_number_of_workers == 8
    assert executor.cluster == mock_cluster
