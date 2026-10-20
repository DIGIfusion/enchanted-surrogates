import os
import subprocess
import sys
import pytest
from enchanted_surrogates.executors.dask_executor import DaskExecutor
from enchanted_surrogates.executors.dask_worker_pool_run import build_worker_job_script


def test_worker_job_script_waits_for_scheduler_file():
    script = build_worker_job_script(
        {"queue": "small", "cores": 1, "memory": "1GB", "processes": 1, "walltime": "00:05:00"},
        "/some/scheduler.json", index=3,
    )
    assert "--scheduler-file /some/scheduler.json" in script
    assert "tcp://" not in script
    assert "--death-timeout 60 " not in script + " "  # must not give up after the default 60 s
    assert "enc_dask_worker-3" in script


def test_external_workers_requires_scheduler_file():
    with pytest.raises(AssertionError):
        DaskExecutor(external_workers=True, SLURMcluster_config={}, scale_n_jobs=1)


def test_worker_started_before_scheduler_connects(tmp_path):
    # the worker is started first and waits for the scheduler file, as a queued SLURM worker job would
    sched_file = str(tmp_path / "scheduler.json")
    worker = subprocess.Popen([
        sys.executable, "-m", "distributed.cli.dask_worker", "--scheduler-file", sched_file,
        "--nthreads", "1", "--death-timeout", "60", "--no-dashboard",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        executor = DaskExecutor(
            external_workers=True, scheduler_file=sched_file, external_workers_timeout=60,
            SLURMcluster_config={"cores": 1, "memory": "1GB", "processes": 1, "queue": "q", "interface": "lo"},
            scale_n_jobs=1,
        )
        executor.start_cluster()
        assert len(executor.client.scheduler_info()["workers"]) == 1
        assert executor.client.submit(lambda x: x + 1, 1).result(timeout=60) == 2
        executor.clean()
    finally:
        worker.terminate()


def _write_config(tmp_path, n_jobs=2, n_executors=1):
    import yaml
    executors = {
        f"e{k}": {
            "type": "DaskExecutor", "external_workers": True,
            "scheduler_file": str(tmp_path / f"pool{k}" / "scheduler.json"),
            "SLURMcluster_config": {"queue": "small", "cores": 1, "memory": "1GB", "processes": 1, "walltime": "00:05:00"},
            "scale_n_jobs": n_jobs,
        } for k in range(n_executors)
    }
    cfg = {"executors": executors, "supervisor": {"base_run_dir": str(tmp_path / "run")}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    main = tmp_path / "main.sbatch"
    main.write_text("#!/bin/bash\n#SBATCH -p small\n#SBATCH -t 01:00:00\npython run.py -cf config.yaml\n")
    return str(path), str(main)


@pytest.mark.parametrize("mode", ["first_any", "each_any", "all"])
def test_modes_submission(tmp_path, monkeypatch, mode):
    from enchanted_surrogates.executors import dask_worker_pool_run as pool
    submitted = []
    monkeypatch.setattr(pool, "sbatch", lambda args: submitted.append(args) or str(100 + len(submitted)))
    cfg, main = _write_config(tmp_path, n_jobs=2, n_executors=2)
    pool.main(["-cf", cfg, "--main-job-script", main, "--mode", mode])

    workers = [a for a in submitted if os.path.basename(a[-1]).startswith("worker_")]
    mains = [a for a in submitted if a[-1] == main]
    if mode == "all":         # one heterogeneous job: main + 4 workers
        assert len(submitted) == 1 and not workers
        het = open(submitted[0][0]).read()
        assert het.count("#SBATCH hetjob") == 4
        assert het.count("srun --het-group=") == 4 and het.count("--overlap") == 4
        assert het.index("srun --het-group=") < het.index("python run.py")
        body = open(os.path.join(os.path.dirname(submitted[0][0]), "e1_worker_1_body.sh")).read()
        assert "ENC_HET_GROUP=4" in body and "srun_wrapper" in body and "distributed.cli.dask_worker" in body
        wrapper = open(os.path.join(os.path.dirname(submitted[0][0]), "srun_wrapper", "srun")).read()
        assert "--het-group=${ENC_HET_GROUP:?}" in wrapper
        return
    assert len(workers) == 4 and len(mains) == 1      # ordinary separate worker jobs
    if mode == "first_any":   # workers 101..104, e0 = 101, 102
        assert mains[0][0] == "--dependency=after:101?after:102"
    else:                     # main submitted first and held, workers release it
        assert submitted[0] == ["--hold", main]
        script = open(workers[0][0]).read()
        assert "scontrol release 101" in script
        assert script.index("scontrol release") < script.index("distributed.cli.dask_worker")


def test_checkin_releases_only_when_every_executor_started(tmp_path):
    from enchanted_surrogates.executors.dask_worker_pool_run import checkin_prologue
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "scontrol").write_text(f"#!/bin/bash\necho \"$@\" >> {tmp_path}/released\n")
    (fake / "scontrol").chmod(0o755)
    checkin = tmp_path / "checkin"

    def start_worker(executor, job_id):
        lines = checkin_prologue(str(checkin), executor, ["e0", "e1"], "999")
        subprocess.run(["bash", "-c", "\n".join(lines)], check=True,
                       env={"PATH": f"{fake}:/usr/bin:/bin", "SLURM_JOB_ID": job_id})

    start_worker("e0", "1")
    start_worker("e0", "2")
    assert not (tmp_path / "released").exists()          # only e0 has started
    start_worker("e1", "3")
    assert (tmp_path / "released").read_text().strip() == "release 999"
