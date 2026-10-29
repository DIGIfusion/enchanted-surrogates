"""main_on_worker_takeover with plain bash processes standing in for the worker jobs, the dask worker and run.py."""
import os
import signal
import subprocess
import time

import pytest
import yaml

from enchanted_surrogates.executors.dask_worker_pool_run import write_takeover_files, wrap_worker_script

STALE, POLL, MAX_TAKEOVERS = 3, 1, 2


def wait_for(condition, timeout=40, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


def lines(path):
    return open(path).read().splitlines() if os.path.exists(path) else []


class Pool:
    def __init__(self, tmp_path, behaviour="run"):
        self.dir = tmp_path
        self.sf = str(tmp_path / "scheduler.json")
        self.cfg = str(tmp_path / "config.yaml")
        self.procs = []
        with open(self.cfg, "w") as f:
            yaml.safe_dump({"supervisor": {"run_mode": "fresh", "base_run_dir": str(tmp_path / "run")}}, f)
        self.set_behaviour(behaviour)
        # fake run.py: records the attempt, writes the scheduler file like the dask scheduler would, then behaves as told
        (tmp_path / "fake_run.py").write_text(
            f'#!/bin/bash\necho "main config=$2" >> {tmp_path}/mains.log\necho started\n'
            f'echo fake-scheduler-$RANDOM > {self.sf}\n'
            f'case "$(cat {tmp_path}/behaviour)" in\n run) sleep 1000;;\n exit0) sleep 2; exit 0;;\n fail) exit 3;;\nesac\n')
        (tmp_path / "fake_dask_worker.sh").write_text(
            f'#!/bin/bash\necho "worker $SLURM_JOB_ID" >> {tmp_path}/workers.log\nsleep 1000\n')
        main_script = (f"#!/bin/bash\n#SBATCH --time=01:00:00\ncd {tmp_path}\n"
                       f"bash {tmp_path}/fake_run.py --config_file {self.cfg} > {tmp_path}/run_out.txt 2>&1\n")
        self.agent = write_takeover_files(str(tmp_path), main_script, self.cfg, [self.sf],
                                          stale=STALE, poll=POLL, max_takeovers=MAX_TAKEOVERS, heartbeat=1)
        self.worker_script = wrap_worker_script(
            f"#!/bin/bash\n#SBATCH -J w\nexport A=1\nbash {tmp_path}/fake_dask_worker.sh\n", self.agent, self.sf)

    def set_behaviour(self, behaviour):
        (self.dir / "behaviour").write_text(behaviour)

    def start_worker_jobs(self, n):
        for i in range(n):
            path = self.dir / f"worker_{i}.sh"
            path.write_text(self.worker_script)
            self.procs.append(subprocess.Popen(["bash", str(path)], env={**os.environ, "SLURM_JOB_ID": str(i)},
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))

    def kill_main(self, attempt):
        """what the end of the host job does: everything of that main process dies at once"""
        pid = int(subprocess.run(["pgrep", "-f", f"main_wrapper.sh {attempt}"], capture_output=True, text=True,
                                 check=True).stdout.split()[0])
        os.killpg(os.getpgid(pid), signal.SIGKILL)

    def cleanup(self):
        for p in self.procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        subprocess.run(["pkill", "-9", "-f", str(self.dir)])


@pytest.fixture
def make_pool(tmp_path):
    pools = []

    def make(behaviour="run"):
        pools.append(Pool(tmp_path, behaviour))
        return pools[-1]
    yield make
    for pool in pools:
        pool.cleanup()


def test_only_one_of_many_workers_starts_the_main_process(make_pool):
    pool = make_pool()
    pool.start_worker_jobs(5)
    wait_for(lambda: len(lines(pool.dir / "workers.log")) >= 5, what="all 5 workers connected")
    time.sleep(STALE + 3)  # long enough for a second main process to appear, if the election were broken
    assert len(lines(pool.dir / "mains.log")) == 1
    assert sorted(os.listdir(pool.dir)).count("main_lock_0") == 1 and not (pool.dir / "main_lock_1").exists()


def test_another_worker_takes_over_when_the_main_process_dies(make_pool):
    pool = make_pool()
    pool.start_worker_jobs(3)
    wait_for(lambda: len(lines(pool.dir / "workers.log")) >= 3, what="3 workers")
    first_sf = (pool.dir / "scheduler.json").read_text()
    pool.kill_main(0)
    wait_for(lambda: len(lines(pool.dir / "mains.log")) >= 2, what="the takeover")
    time.sleep(2)
    mains = lines(pool.dir / "mains.log")
    assert len(mains) == 2, "exactly one worker takes over"
    assert mains[0].endswith("config.yaml") and mains[1].endswith("takeover_resume_config.yaml")
    assert yaml.safe_load(open(pool.dir / "takeover_resume_config.yaml"))["supervisor"]["run_mode"] == "resume"
    assert (pool.dir / "scheduler.json").read_text() != first_sf, "the new main process writes a new scheduler file"
    # every worker reconnects: 3 first starts + 3 restarts
    wait_for(lambda: len(lines(pool.dir / "workers.log")) >= 6, what="workers restarted against the new scheduler")
    assert lines(pool.dir / "run_out.txt").count("started") == 2, "the output of the first attempt is kept"


def test_all_workers_stop_when_the_main_process_finishes(make_pool):
    pool = make_pool("exit0")
    pool.start_worker_jobs(3)
    wait_for(lambda: (pool.dir / "main_done").exists(), what="main_done")
    wait_for(lambda: all(p.poll() is not None for p in pool.procs), what="worker jobs ending")
    assert len(lines(pool.dir / "mains.log")) == 1


def test_gives_up_when_the_main_process_keeps_failing(make_pool):
    pool = make_pool("fail")
    pool.start_worker_jobs(3)
    wait_for(lambda: (pool.dir / "main_gave_up").exists(), what="giving up")
    wait_for(lambda: all(p.poll() is not None for p in pool.procs), what="worker jobs ending")
    assert len(lines(pool.dir / "mains.log")) == 1 + MAX_TAKEOVERS


def test_launch_cleans_the_state_of_an_earlier_launch(tmp_path):
    for name in ("main_heartbeat", "main_done", "main_gave_up", "main_exit_0"):
        (tmp_path / name).write_text("")
    (tmp_path / "main_lock_0").mkdir()
    cfg = tmp_path / "c.yaml"
    cfg.write_text("supervisor: {run_mode: resume}\n")
    write_takeover_files(str(tmp_path), "#!/bin/bash\npython run.py --config_file %s > out.txt 2>&1\n" % cfg, str(cfg), [])
    assert not [n for n in os.listdir(tmp_path) if n.startswith("main_lock") or n in ("main_heartbeat", "main_done", "main_gave_up", "main_exit_0")]
    assert not (tmp_path / "takeover_resume_config.yaml").exists(), "a config that already resumes is used as it is"
    assert ">> out.txt" in (tmp_path / "main_body_resume.sh").read_text()
