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


# ---- is_tasks_running --------------------------------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import MagicMock
from enchanted_surrogates.executors.base_executor import Executor
from enchanted_surrogates.executors.dask_executor import DaskExecutor, _count_unfinished_tasks


def test_base_executor_cannot_tell_so_it_says_tasks_are_running():
    class Dummy(Executor):
        def execute(self, samples, runner_config):
            pass

        def clean(self):
            pass

    assert Dummy().is_tasks_running() is True


def test_count_unfinished_tasks_counts_the_unfinished_states_only():
    states = ["waiting", "queued", "processing", "no-worker", "constrained", "memory", "released", "erred", "forgotten"]
    scheduler = SimpleNamespace(tasks={i: SimpleNamespace(state=s) for i, s in enumerate(states)})
    assert _count_unfinished_tasks(scheduler) == 5


def test_dask_is_tasks_running_follows_the_scheduler():
    executor = DaskExecutor(LocalCluster_config={"n_workers": 1})
    assert executor.is_tasks_running() is False        # the cluster was never started

    executor.client = MagicMock()
    executor.client.run_on_scheduler.return_value = 3
    assert executor.is_tasks_running() is True
    executor.client.run_on_scheduler.return_value = 0
    assert executor.is_tasks_running() is False

    executor.client.run_on_scheduler.side_effect = RuntimeError("scheduler unreachable")
    assert executor.is_tasks_running() is True          # cannot tell: assume running

    executor.is_closed = True
    assert executor.is_tasks_running() is False


def test_dask_is_tasks_running_on_a_real_local_cluster(tmp_path):
    import time
    from dask.distributed import Client, LocalCluster, fire_and_forget
    executor = DaskExecutor(LocalCluster_config={"n_workers": 1})
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None)
    executor.cluster, executor.client = cluster, Client(cluster)
    try:
        assert executor.is_tasks_running() is False
        fire_and_forget(executor.client.submit(time.sleep, 3, pure=False))
        time.sleep(0.5)
        assert executor.is_tasks_running() is True
        for _ in range(100):
            if not executor.is_tasks_running():
                break
            time.sleep(0.2)
        assert executor.is_tasks_running() is False
    finally:
        executor.client.close()
        cluster.close()
