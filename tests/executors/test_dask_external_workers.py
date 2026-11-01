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
    if mode == "each_any":    # the main job is submitted like the workers: not held, no dependency, and it waits for the check-ins itself
        derived = [a for a in submitted if os.path.basename(a[-1]) == "main_each_any.sbatch"]
        assert len(workers) == 4 and len(derived) == 1 and not mains
        assert derived[0] == [derived[0][-1]]                  # no --hold, no --dependency
        script = open(derived[0][-1]).read()
        assert "#SBATCH -p small" in script and "#SBATCH -t 01:00:00" in script
        assert script.index("#SBATCH -t") < script.index("until ") < script.index("python run.py")
        assert "e0.*" in script and "e1.*" in script
        worker = open(workers[0][0]).read()
        assert "touch" in worker and "scontrol" not in worker
        assert worker.index("touch") < worker.index("distributed.cli.dask_worker")
        return
    assert len(workers) == 4 and len(mains) == 1      # ordinary separate worker jobs
    assert mains[0][0] == "--dependency=after:101?after:102"   # first_any: workers 101..104, e0 = 101, 102


def test_main_job_waits_until_every_executor_has_checked_in(tmp_path):
    from enchanted_surrogates.executors.dask_worker_pool_run import checkin_prologue, wait_for_checkins
    checkin = tmp_path / "checkin"

    def start_worker(executor, job_id):
        subprocess.run(["bash", "-c", "\n".join(checkin_prologue(str(checkin), executor))], check=True,
                       env={"PATH": "/usr/bin:/bin", "SLURM_JOB_ID": job_id})

    def main_job_gets_through():
        try:
            subprocess.run(["bash", "-c", "\n".join(wait_for_checkins(str(checkin), ["e0", "e1"], poll=1))],
                           check=True, timeout=3, env={"PATH": "/usr/bin:/bin"})
            return True
        except subprocess.TimeoutExpired:
            return False

    assert not main_job_gets_through()           # nobody has started
    start_worker("e0", "1")
    start_worker("e0", "2")
    assert not main_job_gets_through()           # only e0 has started
    start_worker("e1", "3")
    assert main_job_gets_through()
