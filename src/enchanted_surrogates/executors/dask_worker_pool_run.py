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
    each_any   once at least one worker job of EVERY external executor has started.
               SLURM cannot express this as a dependency, so the main job is
               submitted held and each worker job, when it starts, checks in and
               releases the main job (scontrol release) when every executor has
               checked in.
    all        everything at once: the main process and all workers form one
               heterogeneous job (#SBATCH hetjob), so no worker waits idle for the
               main job. Each worker runs as `srun --het-group=k --overlap` on its own
               component, with an `srun` wrapper first on its PATH that adds
               --het-group=k, so a runner that calls srun itself (e.g. GENE) lands on
               the worker's own nodes. The job ends when the main process ends.
"""
import argparse
import os
import shutil
import subprocess

from dask_jobqueue.slurm import SLURMJob

from enchanted_surrogates.utils.config_helpers import load_configuration

WAIT_FOREVER = 60 * 60 * 24 * 30  # seconds a worker waits for the scheduler file
MODES = ("first_any", "each_any", "all")


def build_worker_job_script(slurm_config: dict, scheduler_file: str, index: int = 0, prologue: list[str] = ()) -> str:
    """Returns a SLURM job script for one dask worker job that connects through scheduler_file.
    `prologue` lines run first in the job, before the configured job_script_prologue."""
    config = {
        k: v for k, v in slurm_config.items()
        if k not in ("scheduler_options", "n_workers", "silence_logs")
    }
    config.setdefault("job_name", "enc_dask_worker")
    config.setdefault("death_timeout", WAIT_FOREVER)  # default 60 s would end the wait for the scheduler
    config["job_script_prologue"] = list(prologue) + list(config.get("job_script_prologue", []))
    placeholder = "tcp://scheduler-from-file:0"
    job = SLURMJob(scheduler=placeholder, name=f"enc_dask_worker-{index}", **config)
    script = job.job_script()
    assert placeholder in script
    return script.replace(placeholder, f"--scheduler-file {scheduler_file}")


def checkin_prologue(checkin_dir: str, executor: str, all_executors: list[str], main_job_id: str) -> list[str]:
    """Lines for a worker job: record that it started, and release the held main job
    once at least one worker of every executor has started."""
    have_all = " && ".join(f"ls {checkin_dir}/{e}.* >/dev/null 2>&1" for e in all_executors)
    return [
        f"mkdir -p {checkin_dir} && touch {checkin_dir}/{executor}.$SLURM_JOB_ID",
        f"if {have_all}; then scontrol release {main_job_id} || true; fi",
    ]


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
    args = parser.parse_args(argv)

    config = load_configuration(args.config_file)
    executors = {name: cfg for name, cfg in config.executors.items() if cfg.get("external_workers")}
    if not executors:
        raise SystemExit("No executor with external_workers: true in the config")
    first_dir = os.path.dirname(os.path.abspath(next(iter(executors.values()))["scheduler_file"]))

    # each_any: the main job is submitted first, held, so the workers know which job to release
    main_id, prologue_for = None, {}
    if args.mode == "each_any":
        main_id = "MAINJOBID" if args.dry_run else sbatch(["--hold", args.main_job_script])
        if not args.dry_run:
            print(f"submitted main job {main_id} (held until a worker of every executor has started)")
        checkin_dir = os.path.join(first_dir, f"checkin_{main_id}")
        prologue_for = {name: checkin_prologue(checkin_dir, name, list(executors), main_id) for name in executors}

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
                f.write(build_worker_job_script(slurm_config, scheduler_file, i, prologue_for.get(name, [])))
            if args.mode == "all":
                het_workers.append((f"{name}_worker_{i}", path))
            elif args.dry_run:
                print(f"wrote {path}")
            else:
                try:
                    ids[name].append(sbatch([path]))
                except subprocess.CalledProcessError as e:
                    if main_id:
                        print(f"WARNING: worker submission failed, main job {main_id} stays held: scancel {main_id}")
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
    if args.dry_run or args.mode == "each_any":
        return
    dependency = "--dependency=" + "?".join(f"after:{j}" for j in ids[next(iter(executors))])
    main_id = sbatch([dependency, "--kill-on-invalid-dep=yes", args.main_job_script])
    print(f"submitted main job {main_id} ({dependency})")


if __name__ == "__main__":
    main()
