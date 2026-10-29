"""
Alternative way to start a run: the Dask worker jobs are submitted *before* the
main process exists, so the main process (the supervisor loop, run.py) does not
use up its walltime while the workers wait, possibly for days, in the queue.

Run from the login node:

    python -m enchanted_surrogates.executors.dask_worker_pool_run -cf config.yaml \\
        --main-job-script main.sbatch [--mode first_any|each_any|all] [--dry-run]

`main.sbatch` is an ordinary sbatch script that runs run.py with the same config.
Every executor with `external_workers: true` in the config gets its workers from
here; they wait for that executor's `scheduler_file` (dask `--scheduler-file`),
which the main process writes when it starts the scheduler.

All workers are ordinary, separate SLURM jobs. The modes only decide when the
main job may start:
    first_any  once ANY worker job of the first external executor has started
               (--dependency=after:w1?after:w2?...)
    each_any   once at least one worker job of EVERY external executor has started, and
               the main job itself has started too: the main job is submitted at once, like
               the workers, so it queues in parallel with them and does NOT wait in the queue
               after the workers have started. Its script is the given one with a wait inserted
               after the #SBATCH lines: it polls the check-in files that each worker job writes
               when it starts (<pool dir>/checkin_each_any/<executor>.<job id>) until every
               executor has one, and only then runs the script's commands. The walltime of the
               main job must therefore cover this wait as well as the run (the main job is
               small, so waiting costs little).
    all        everything at once: the main process and all workers form one
               heterogeneous job (#SBATCH hetjob), so no worker waits idle for the
               main job. Each worker runs as `srun --het-group=k --overlap` on its own
               component, with an `srun` wrapper first on its PATH that adds
               --het-group=k, so a runner that calls srun itself (e.g. GENE) lands on
               the worker's own nodes. The job ends when the main process ends.
    main_on_worker
               no separate main job: the first worker job of any executor to start also
               runs the main process, in the background on its own node, next to its
               dask worker. The workers race for the right to do so with an atomic
               `mkdir <pool dir>/main_on_worker_lock`; this stale lock is removed when
               this script submits the workers. The main process ends with the job of the
               worker that hosts it (walltime, node failure); a run can then be continued
               with `run_mode: resume`. Its output goes where the main script sends it.
    main_on_worker_takeover
               like main_on_worker, but when the main process dies (its host job hits the
               walltime, the node fails, run.py crashes) another worker job takes over:
               every worker job runs a watchdog, the main process touches a heartbeat file,
               and when the heartbeat is stale for `--takeover-stale` seconds one worker
               (atomic `mkdir main_lock_<attempt>`) starts the main process again with
               `run_mode: resume`. The dask workers reconnect to the new scheduler. Gives up
               after `--takeover-max` takeovers, and stops when run.py exits with code 0
               (`main_done`). See `write_takeover_files`.
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import yaml

from dask_jobqueue.slurm import SLURMJob

from enchanted_surrogates.utils.config_helpers import load_configuration

WAIT_FOREVER = 60 * 60 * 24 * 30  # seconds a worker waits for the scheduler file
TAKEOVER = "main_on_worker_takeover"
MODES = ("first_any", "each_any", "all", "main_on_worker", TAKEOVER)


def build_worker_job_script(slurm_config: dict, scheduler_file: str, index: int = 0, prologue: list[str] = ()) -> str:
    """Returns a SLURM job script for one dask worker job that connects through scheduler_file.
    `prologue` lines run first in the job, before the configured job_script_prologue."""
    config = {
        k: v for k, v in slurm_config.items()
        if k not in ("scheduler_options", "n_workers", "silence_logs", "name")  # name: the cluster's name, the worker job gets its own
    }
    config.setdefault("job_name", "enc_dask_worker")
    config.setdefault("death_timeout", WAIT_FOREVER)  # default 60 s would end the wait for the scheduler
    config["job_script_prologue"] = list(prologue) + list(config.get("job_script_prologue", []))
    placeholder = "tcp://scheduler-from-file:0"
    job = SLURMJob(scheduler=placeholder, name=f"enc_dask_worker-{index}", **config)
    script = job.job_script()
    assert placeholder in script
    return script.replace(placeholder, f"--scheduler-file {scheduler_file}")


def checkin_prologue(checkin_dir: str, executor: str) -> list[str]:
    """Lines for a worker job: record that it started (each_any: the main job waits for these files)."""
    return [f"mkdir -p {checkin_dir} && touch {checkin_dir}/{executor}.$SLURM_JOB_ID"]


def wait_for_checkins(checkin_dir: str, executors: list[str], poll: int = 10) -> list[str]:
    """Lines for the main job: wait until a worker job of every executor has checked in."""
    have_all = " && ".join(f"ls {checkin_dir}/{e}.* >/dev/null 2>&1" for e in executors)
    return [f"until {have_all}; do sleep {poll}; done"]


def build_each_any_main_script(script: str, checkin_dir: str, executors: list[str], poll: int = 10) -> str:
    """The main job script with the wait for the check-ins of every executor after its #SBATCH lines."""
    directives, body = split_script(script)
    wait = ["# each_any: wait until a worker job of every external executor has started (they check in with a file)"]
    return "\n".join(["#!/bin/bash"] + directives + wait + wait_for_checkins(checkin_dir, executors, poll) + body) + "\n"


def main_on_worker_prologue(lock_dir: str, main_body_path: str, log_path: str) -> list[str]:
    """Lines for a worker job: the first worker job to get here (mkdir is atomic, also on Lustre)
    starts the main process in the background. The others skip it, whatever the state of base_run_dir."""
    return [f"if mkdir {lock_dir} 2>/dev/null; then nohup bash {main_body_path} > {log_path}-$SLURM_JOB_ID.out 2>&1 & fi"]


def split_script(script: str) -> tuple[list[str], list[str]]:
    """(#SBATCH lines, remaining body lines) of a job script, shebang dropped"""
    directives, body = [], []
    for line in script.splitlines():
        if line.startswith("#!"):
            continue
        (directives if line.strip().startswith("#SBATCH") else body).append(line)
    return directives, body


def write_executable(path: str, text: str):
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, 0o755)


MAIN_WRAPPER = r"""#!/bin/bash
# runs the main process of attempt $1 with a heartbeat; started by the takeover agent in its own session
n=$1
POOL=@POOL@
touch $POOL/main_heartbeat
( while true; do touch $POOL/main_heartbeat; sleep @HEARTBEAT@; done ) &
hb=$!
if [ "$n" = 0 ]; then bash $POOL/main_body.sh; else bash $POOL/main_body_resume.sh; fi
rc=$?
kill $hb 2>/dev/null
echo $rc > $POOL/main_exit_$n
[ $rc -eq 0 ] && touch $POOL/main_done
exit $rc
"""

# sourced at the end of a worker job script, which defines WORKER_CMD (the dask-worker command) and WORKER_SF
TAKEOVER_AGENT = r"""POOL=@POOL@
STALE=@STALE@
POLL=@POLL@
MAX_TAKEOVERS=@MAX_TAKEOVERS@
SCHEDULER_FILES=(@SCHEDULER_FILES@)
HB=$POOL/main_heartbeat
DONE=$POOL/main_done
GAVE_UP=$POOL/main_gave_up

mtime() { stat -c %Y "$1" 2>/dev/null || echo 0; }
finished() { [ -e $DONE ] || [ -e $GAVE_UP ]; }

next_attempt() { local n=0; while [ -d $POOL/main_lock_$n ]; do n=$((n+1)); done; echo $n; }

# the main process is considered dead when neither its heartbeat nor the newest attempt lock is fresh.
# The attempt number is read BEFORE the age: a lock created in between is then either not yet counted
# (and mkdir of the same number fails) or already counted and fresh (so we see no stale main process).
maybe_take_over() {
    finished && return
    local n=$(next_attempt) ref m l
    ref=$(mtime $HB)
    for l in $POOL/main_lock_*; do [ -d "$l" ] || continue; m=$(mtime $l); [ $m -gt $ref ] && ref=$m; done
    [ $(( $(date +%s) - ref )) -le $STALE ] && return
    if [ $n -gt $MAX_TAKEOVERS ]; then touch $GAVE_UP; return; fi
    if mkdir $POOL/main_lock_$n 2>/dev/null; then       # atomic: one worker wins
        rm -f "${SCHEDULER_FILES[@]}"                    # the stale scheduler address of the dead main process
        setsid nohup bash $POOL/main_wrapper.sh $n > $POOL/main_attempt_$n.out 2>&1 &
    fi
}

watchdog() { while ! finished; do maybe_take_over; sleep $POLL; done; }

stop_worker() { kill $1 2>/dev/null; for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $1 2>/dev/null || return; sleep 1; done; kill -9 $1 2>/dev/null; }

# keeps one dask worker connected to the current scheduler: waits for a scheduler file, and restarts the
# worker when the scheduler it connected to is replaced (a takeover) or gone
worker_loop() {
    while ! finished; do
        until [ -e "$WORKER_SF" ]; do finished && return; sleep $POLL; done
        local m=$(mtime "$WORKER_SF")
        ( eval "exec $WORKER_CMD" ) &
        local pid=$!
        while kill -0 $pid 2>/dev/null; do
            if finished || [ ! -e "$WORKER_SF" ] || [ "$(mtime "$WORKER_SF")" != "$m" ]; then stop_worker $pid; break; fi
            sleep $POLL
        done
        wait $pid 2>/dev/null
        sleep $POLL
    done
}

watchdog &
worker_loop
"""


def _fill(template: str, **values) -> str:
    for k, v in values.items():
        template = template.replace(f"@{k.upper()}@", str(v))
    return template


def write_takeover_files(pool_dir: str, main_script: str, config_file: str, scheduler_files: list[str],
                         stale: int = 300, poll: int = 30, max_takeovers: int = 3, heartbeat: int = 30,
                         clean: bool = True) -> str:
    """Writes the files of mode main_on_worker_takeover into pool_dir and returns the path of the agent
    script that every worker job sources. Attempt 0 runs the main script's commands; a takeover runs them
    with a copy of the config that has `run_mode: resume` and appends instead of truncating the output file.
    With `clean`, the state of an earlier launch (locks, heartbeat, markers) is removed."""
    os.makedirs(pool_dir, exist_ok=True)
    if clean:
        for name in os.listdir(pool_dir):
            path = os.path.join(pool_dir, name)
            if name.startswith("main_lock_") and os.path.isdir(path):
                os.rmdir(path)
            elif name in ("main_heartbeat", "main_done", "main_gave_up") or name.startswith("main_exit_"):
                os.remove(path)

    _, body = split_script(main_script)
    write_executable(os.path.join(pool_dir, "main_body.sh"), "\n".join(["#!/bin/bash"] + body) + "\n")

    config_file = os.path.abspath(config_file)
    resume_config = config_file
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("supervisor", {}).get("run_mode") != "resume":
        cfg.setdefault("supervisor", {})["run_mode"] = "resume"
        resume_config = os.path.join(pool_dir, "takeover_resume_config.yaml")
        with open(resume_config, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    resume_body = []
    for line in body:
        if "run.py" in line:
            line = line.replace(config_file, resume_config)
            line = re.sub(r"(\s)>(\s)", r"\1>>\2", line)  # keep the output of the earlier attempts
        resume_body.append(line)
    write_executable(os.path.join(pool_dir, "main_body_resume.sh"), "\n".join(["#!/bin/bash"] + resume_body) + "\n")

    write_executable(os.path.join(pool_dir, "main_wrapper.sh"), _fill(MAIN_WRAPPER, pool=pool_dir, heartbeat=heartbeat))
    agent = os.path.join(pool_dir, "takeover_agent.sh")
    write_executable(agent, _fill(TAKEOVER_AGENT, pool=pool_dir, stale=stale, poll=poll, max_takeovers=max_takeovers,
                                  scheduler_files=" ".join(shlex.quote(f) for f in scheduler_files)))
    return agent


def wrap_worker_script(script: str, agent_path: str, scheduler_file: str) -> str:
    """The worker job script with its last line, the dask-worker command, handed to the takeover agent."""
    lines = script.rstrip("\n").split("\n")
    assert "dask_worker" in lines[-1], f"expected the dask-worker command as last line, got: {lines[-1]}"
    return "\n".join(lines[:-1] + [
        f"WORKER_CMD={shlex.quote(lines[-1])}", f"WORKER_SF={shlex.quote(scheduler_file)}", f"source {agent_path}", ""])



def build_het_script(main_script: str, workers: list[tuple[str, str]], het_dir: str) -> str:
    """One heterogeneous job: component 0 runs the main script body, component k>=1
    runs worker k-1. workers: [(name, worker job script), ...]"""
    real_srun = shutil.which("srun") or "/usr/bin/srun"
    wrapper_dir = os.path.join(het_dir, "srun_wrapper")
    os.makedirs(wrapper_dir, exist_ok=True)
    # inside a heterogeneous job srun must name its component, a plain srun would go to component 0
    write_executable(os.path.join(wrapper_dir, "srun"),
                     f'#!/bin/bash\nexec {real_srun} --het-group=${{ENC_HET_GROUP:?}} "$@"\n')

    main_directives, main_body = split_script(main_script)
    lines = ["#!/bin/bash"] + main_directives
    launch = []
    for k, (name, script) in enumerate(workers, start=1):
        directives, body = split_script(script)
        # job name and output files belong to the whole job (component 0)
        directives = [d for d in directives if d.split()[1].split("=")[0] not in ("-J", "--job-name", "-o", "--output", "-e", "--error")]
        lines += ["#SBATCH hetjob"] + directives
        body_path = os.path.join(het_dir, f"{name}_body.sh")
        write_executable(body_path, "\n".join([
            "#!/bin/bash",
            # make the step look like the worker's own job to the runner
            f"export ENC_HET_GROUP={k} PATH={wrapper_dir}:$PATH",
            f'[ -n "${{SLURM_NTASKS_HET_GROUP_{k}:-}}" ] && export SLURM_NTASKS=$SLURM_NTASKS_HET_GROUP_{k}',
            f'[ -n "${{SLURM_TASKS_PER_NODE_HET_GROUP_{k}:-}}" ] && export SLURM_NTASKS_PER_NODE=${{SLURM_TASKS_PER_NODE_HET_GROUP_{k}%%(*}}',
        ] + body) + "\n")
        log_path = os.path.join(het_dir, "logs", f"{name}-%j.out")
        # --overlap: the worker's own step must not hold the resources the runner's srun needs;
        # SLURM_MEM_PER_*: the main component's memory option would clash with the worker's
        launch.append(f"env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_GPU -u SLURM_MEM_PER_NODE "
                      f"srun --het-group={k} --ntasks=1 --overlap --output={log_path} {body_path} &")
    return "\n".join(lines + ["", "# workers, one per heterogeneous component"] + launch + ["", "# main process"] + main_body) + "\n"


def sbatch(args: list[str]) -> str:
    out = subprocess.run(["sbatch", "--parsable"] + args, check=True, capture_output=True, text=True)
    return out.stdout.strip().split(";")[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Start a run with the dask workers submitted before the main process")
    parser.add_argument("-cf", "--config_file", required=True)
    parser.add_argument("--main-job-script", required=True, help="sbatch script that runs run.py with the same config")
    parser.add_argument("--mode", choices=MODES, default="first_any")
    parser.add_argument("--dry-run", action="store_true", help="only write the job scripts")
    parser.add_argument("--takeover-stale", type=int, default=300, help="main_on_worker_takeover: seconds without a heartbeat after which a worker takes over")
    parser.add_argument("--takeover-poll", type=int, default=30, help="main_on_worker_takeover: seconds between the checks of a worker's watchdog")
    parser.add_argument("--takeover-max", type=int, default=3, help="main_on_worker_takeover: give up after this many takeovers")
    args = parser.parse_args(argv)

    config = load_configuration(args.config_file)
    executors = {name: cfg for name, cfg in config.executors.items() if cfg.get("external_workers")}
    if not executors:
        raise SystemExit("No executor with external_workers: true in the config")
    first_dir = os.path.dirname(os.path.abspath(next(iter(executors.values()))["scheduler_file"]))

    # each_any: the workers check in when they start; the main job (submitted with them, not held) waits for the check-ins itself
    prologue_for = {}
    checkin_dir = os.path.join(first_dir, "checkin_each_any")
    if args.mode == "each_any":
        if not args.dry_run:
            shutil.rmtree(checkin_dir, ignore_errors=True)   # check-ins of an earlier launch would end the wait at once
        prologue_for = {name: checkin_prologue(checkin_dir, name) for name in executors}

    if args.mode == "main_on_worker":
        lock_dir = os.path.join(first_dir, "main_on_worker_lock")
        main_body_path = os.path.join(first_dir, "main_on_worker_body.sh")
        if not args.dry_run:
            if os.path.isdir(lock_dir):
                os.rmdir(lock_dir)  # a stale lock from an earlier launch would stop every worker from starting the main process
            _, main_body = split_script(open(args.main_job_script).read())
            write_executable(main_body_path, "\n".join(["#!/bin/bash"] + main_body) + "\n")
        mow_prologue = main_on_worker_prologue(lock_dir, main_body_path, os.path.join(first_dir, "main_on_worker"))
        prologue_for = {name: mow_prologue for name in executors}

    agent = None
    if args.mode == TAKEOVER:
        agent = write_takeover_files(
            first_dir, open(args.main_job_script).read(), args.config_file,
            [os.path.abspath(c["scheduler_file"]) for c in executors.values()],
            stale=args.takeover_stale, poll=args.takeover_poll, max_takeovers=args.takeover_max,
            heartbeat=max(1, min(30, args.takeover_stale // 4)), clean=not args.dry_run)

    ids = {name: [] for name in executors}
    het_workers = []
    for name, cfg in executors.items():
        scheduler_file = os.path.abspath(cfg["scheduler_file"])
        pool_dir = os.path.join(os.path.dirname(scheduler_file), f"worker_pool_{name}")
        os.makedirs(pool_dir, exist_ok=True)
        if os.path.exists(scheduler_file) and not args.dry_run:
            os.remove(scheduler_file)  # a stale file would send the workers to a dead scheduler
        slurm_config = dict(cfg["SLURMcluster_config"])
        slurm_config.setdefault("log_directory", os.path.join(pool_dir, "logs"))
        for i in range(int(cfg["scale_n_jobs"])):
            path = os.path.join(pool_dir, f"worker_{i}.sh")
            with open(path, "w") as f:
                text = build_worker_job_script(slurm_config, scheduler_file, i, prologue_for.get(name, []))
                f.write(wrap_worker_script(text, agent, scheduler_file) if agent else text)
            if args.mode == "all":
                het_workers.append((f"{name}_worker_{i}", path))
            elif args.dry_run:
                print(f"wrote {path}")
            else:
                try:
                    ids[name].append(sbatch([path]))
                except subprocess.CalledProcessError as e:
                    raise SystemExit(e.stderr)
                print(f"submitted worker job {ids[name][-1]} ({name} worker {i})")

    if args.mode == "all":
        het_dir = os.path.join(first_dir, "hetjob")
        os.makedirs(os.path.join(het_dir, "logs"), exist_ok=True)
        het_path = os.path.join(het_dir, "main_hetjob.sh")
        with open(het_path, "w") as f:
            f.write(build_het_script(open(args.main_job_script).read(),
                                     [(n, open(p).read()) for n, p in het_workers], het_dir))
        if args.dry_run:
            print(f"wrote {het_path}")
        else:
            print(f"submitted heterogeneous job {sbatch([het_path])} (main process + {len(het_workers)} workers)")
        return
    if args.mode == "each_any":
        main_path = os.path.join(first_dir, "main_each_any.sbatch")
        with open(main_path, "w") as f:
            f.write(build_each_any_main_script(open(args.main_job_script).read(), checkin_dir, list(executors)))
        if args.dry_run:
            print(f"wrote {main_path}")
        else:
            print(f"submitted main job {sbatch([main_path])} (queues with the workers, waits for a worker of every executor to start)")
        return
    if args.dry_run or args.mode in ("main_on_worker", TAKEOVER):
        if args.mode in ("main_on_worker", TAKEOVER) and not args.dry_run:
            print(f"no main job: the first worker job to start runs {args.main_job_script}'s commands in the background")
        return
    dependency = "--dependency=" + "?".join(f"after:{j}" for j in ids[next(iter(executors))])
    main_id = sbatch([dependency, "--kill-on-invalid-dep=yes", args.main_job_script])
    print(f"submitted main job {main_id} ({dependency})")


if __name__ == "__main__":
    main()
