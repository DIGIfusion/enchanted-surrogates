# Starting workers before the main process

On busy HPC systems, GPU worker jobs can wait in the queue for days. With a
normal run, the main process (`run.py`, the supervisor loop that submits
samples, collects results and picks the next samples) creates the Dask scheduler
and then submits the worker jobs itself. If the main process is a SLURM job, its
walltime runs out while it waits for the workers to start.

`dask_worker_pool_run` is an alternative way to start a run that reverses the
order. The worker jobs are submitted first from the login node, and the main
process is only started by SLURM once workers are available. The run itself is
unchanged: same config, same supervisor loop, same output.

## How it works

1. You run `dask_worker_pool_run` on the login node. It submits the worker jobs
   and the main job, then exits.
2. Each worker runs `dask worker --scheduler-file <scheduler_file>`. When it
   starts, it waits for that file to appear.
3. SLURM holds the main job until workers are available. How this is done depends
   on the [mode](#modes).
4. The main job runs `run.py` with the same config. Its `DaskExecutor` starts the
   scheduler, which writes `scheduler_file`, and the waiting workers connect. The
   executor submits no worker jobs of its own.
5. The supervisor loop runs as usual. The workers stay alive between batches
   (e.g. active learning cycles) until the run ends.

## Configuration

Mark each Dask executor that should use pre-submitted workers with
`external_workers: true` and give it a `scheduler_file`:

```yaml
executors:
  e1:
    type: DaskExecutor
    external_workers: true
    scheduler_file: "/path/to/my_study/pool/scheduler.json"
    external_workers_timeout: 600   # optional: seconds the main process waits for the first worker
    SLURMcluster_config:
      account: "project_xxx"
      queue: "standard-g"
      cores: 56
      memory: "400GB"
      processes: 1
      walltime: "2-00:00:00"
      interface: "hsn0"
      job_script_prologue: ['module load ...']
    scale_n_jobs: 2
```

- `SLURMcluster_config` and `scale_n_jobs` mean the same as for a normal Dask
  executor: they define the worker jobs and how many there are.
- `scheduler_file` must be **outside** `base_run_dir`, which must not exist
  before the run starts. The worker job scripts and logs are written next to it,
  in `worker_pool_<executor name>/`.
- Use one `scheduler_file` per executor. Every executor with
  `external_workers: true` in the config gets its workers from
  `dask_worker_pool_run`.
- Adaptive scaling (`scale_n_jobs_min` / `scale_n_jobs_max`) is not supported
  with `external_workers`.

## Main job script

The main job is an ordinary sbatch script that runs `run.py` with the same
config, usually on a small CPU allocation:

```bash
#!/bin/bash
#SBATCH -J es_main
#SBATCH -A project_xxx
#SBATCH -p small
#SBATCH -n 1
#SBATCH --mem-per-cpu=2G
#SBATCH -t 1-00:00:00
#SBATCH -o /path/to/my_study/main-%j.out

source /path/to/.venv/bin/activate
python /path/to/enchanted-surrogates/src/run.py -cf /path/to/config.yaml
```

Its walltime only has to cover the run itself, not the queue wait for the workers. In mode `all` its `#SBATCH` lines become the first component of the heterogeneous job.

## Starting the run

From the login node:

```bash
python -m enchanted_surrogates.executors.dask_worker_pool_run \
    -cf /path/to/config.yaml --main-job-script main.sbatch --mode first_any
```

`--dry-run` writes all job scripts without submitting anything, so you can
check them first.

## Modes

The modes decide when the main process starts:

| Mode | Main process starts when | How |
|---|---|---|
| `first_any` (default) | any worker job of the first external executor has started | workers are separate jobs; the main job has `--dependency=after:w1?after:w2...` |
| `each_any` | at least one worker job of every external executor has started, and the main job itself has started | workers and the main job are all submitted at once, so the main job queues in parallel with the workers and does not wait in the queue after they have started. Its script is yours with a wait after the `#SBATCH` lines: it polls the check-in files that each worker job writes when it starts (`checkin_each_any/<executor>.<job id>` next to the first executor's `scheduler_file`) until every executor has one |
| `main_on_worker` | the first worker job of any executor to start | no separate main job: that worker also runs the main process in the background on its node (atomic `mkdir main_on_worker_lock` picks the host; a stale lock is removed at launch). The main process ends with the host worker's job, so continue with `run_mode: resume` |
| `main_on_worker_takeover` | the first worker job of any executor to start | like `main_on_worker`, and when the main process dies (host job at its walltime, node failure, crash) another worker job takes over: every worker job runs a watchdog, the main process touches `main_heartbeat`, and when it is stale for `--takeover-stale` s (default 300) one worker (atomic `mkdir main_lock_<attempt>`) starts the main process again with `run_mode: resume`, appending to the output file. The dask workers reconnect to the new scheduler. It gives up after `--takeover-max` takeovers (default 3, `main_gave_up`) and stops when run.py exits with code 0 (`main_done`). In-flight runs of a dead main process are lost and rerun by the resume |
| `all` | all resources can start at the same time | the main process and all workers form one heterogeneous job (`#SBATCH hetjob`) |

In `first_any`, workers that start before the main job wait idle for it, and that time
comes out of their walltime; the main job is only queued once a worker has started.
`first_any` gets the main process going soonest. In `each_any` the main job is queued
together with the workers, so it can start before them and then waits (a small job, little
cost) until a worker of every executor has started: the walltime of the main job must cover
that wait as well as the run. `each_any` guarantees that every executor of a sequential
workflow (e.g. HELENA then GENE) has a worker when the first batch is submitted. Workers that
start before the main job wait idle for it.

`all` has no idle workers: everything starts at the same moment, at the cost of
SLURM having to find all the resources at once, which can make the queue wait
longer. With `each_any`, the derived main script is written to `main_each_any.sbatch` next to
the first executor's `scheduler_file`, and the check-in files of an earlier launch are removed
when the workers are submitted.

### How `all` runs the workers

- The `#SBATCH` lines of your main script become component 0; each worker job
  becomes component `k` with its own `#SBATCH` lines (the job name and output
  files belong to component 0).
- Each worker is started with `srun --het-group=<k> --ntasks=1 --overlap`.
  `--overlap` keeps the worker's own step from holding the resources that the
  runner's `srun` needs.
- Inside a heterogeneous job, an `srun` that does not name its component goes to
  component 0. Each worker therefore has a small `srun` wrapper first on its
  `PATH` that adds `--het-group=<k>`, and `SLURM_NTASKS` / `SLURM_NTASKS_PER_NODE`
  are set to the component's values. A runner that calls `srun` itself (e.g.
  GENE: `srun --ntasks $SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE ...`)
  then lands on the worker's own nodes without any change to the runner.
- The memory variables of component 0 (`SLURM_MEM_PER_*`) are cleared for the
  worker steps, since `srun` refuses mixed memory options.
- The job ends when the main process ends. The generated script, the worker
  start scripts, the `srun` wrapper and the worker logs are in `hetjob/` next to
  the first executor's `scheduler_file`.

Heterogeneous jobs that combine partitions (e.g. a CPU main process with GPU
workers) must be allowed by your site; on LUMI `small`/`debug` with
`small-g`/`dev-g` works.

## Things to know

- **Worker walltime** starts when the worker job starts, not when the main
  process connects. In `first_any`, give the workers enough walltime to cover the
  main job's own queue wait.
- **Stale scheduler files** are removed by `dask_worker_pool_run` before
  submitting, so workers do not connect to the scheduler of an old run.
- **Waiting workers** use a 30-day `--death-timeout`, so they do not give up
  while waiting for the scheduler. The default of 60 s would end them early.
- **Ending the run**: when the supervisor finishes, the executor shuts down the
  scheduler and the workers, which ends their jobs.

Example configuration: `configs/example_dask_slurm_external_workers.yaml`.
See also the [Dask Executor](executors/dask_executor.md) reference.
