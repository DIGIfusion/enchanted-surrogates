"""
Standalone stress test for the DaskExecutor.submit_batch future-handling
leak (see fix/dask-executor-future-memory-leak branch). Submits many small,
fast ExampleRunner tasks through a LocalCluster and tracks the CLIENT
process's own RSS over time (LocalCluster's scheduler runs as a background
thread inside this process by default, so its memory shows up here directly
-- no separate process to attach to).

Three future-handling variants are compared, monkeypatched onto
DaskExecutor.submit_batch one at a time so the rest of the pipeline
(runner, packer, supervisor-equivalent submission loop) is identical across
all three runs -- only this one method's body differs:

    baseline            : the current fire_and_forget(new_future) behavior
                           (expected to leak / grow RSS roughly linearly
                           with total tasks submitted).
    done_callback_release : new_future.add_done_callback(lambda f: f.release())
                           -- release the future's scheduler-side state the
                           moment it finishes.
    tracked_list_prune : keep a plain list of futures; every N submissions,
                           drop the ones that are already .done() (letting
                           normal Python refcounting + Dask's non-
                           fire_and_forget future GC release them) -- a more
                           conservative, more "documented API" alternative
                           to calling .release() directly.

Usage::
    python leak_test.py --variant baseline --n-tasks 4000 --batch-size 20
    python leak_test.py --variant done_callback_release --n-tasks 4000 --batch-size 20
    python leak_test.py --variant tracked_list_prune --n-tasks 4000 --batch-size 20

Writes a CSV of (n_submitted, elapsed_s, rss_mb) samples to
results/<variant>.csv in this directory, so the 3 runs can be plotted
together afterward.
"""
import argparse
import gc
import os
import shutil
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from enchanted_surrogates.executors.dask_executor import DaskExecutor
from enchanted_surrogates.executors.simulation_task import run_simulation_task
from enchanted_surrogates.utils.logger import setup_logger
from dask.distributed import fire_and_forget

RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# The 3 variants under test, same signature as
# DaskExecutor.submit_batch's real body (see dask_executor.py).
# ------------------------------------------------------------
def submit_batch_baseline(self, run_dir_sample_pairs, runner_config, base_run_dir=None, client=None):
    if not client:
        client = self.client
    assert client is not None
    for run_dir, sample_params in run_dir_sample_pairs:
        new_future = client.submit(run_simulation_task, runner_config, run_dir, sample_params)
        fire_and_forget(new_future)


def submit_batch_done_callback_release(self, run_dir_sample_pairs, runner_config, base_run_dir=None, client=None):
    if not client:
        client = self.client
    assert client is not None
    for run_dir, sample_params in run_dir_sample_pairs:
        new_future = client.submit(run_simulation_task, runner_config, run_dir, sample_params)
        new_future.add_done_callback(lambda f: f.release())


def submit_batch_tracked_list_prune(self, run_dir_sample_pairs, runner_config, base_run_dir=None, client=None):
    if not client:
        client = self.client
    assert client is not None
    if not hasattr(self, "_tracked_futures"):
        self._tracked_futures = []
    for run_dir, sample_params in run_dir_sample_pairs:
        new_future = client.submit(run_simulation_task, runner_config, run_dir, sample_params)
        # Deliberately NOT fire_and_forget: a plain (non-fire-and-forget)
        # future is released once nothing references it anymore AND its
        # task has completed, matching normal Dask client GC semantics.
        self._tracked_futures.append(new_future)
    # Prune every call: drop completed futures from our own list so their
    # last remaining client-side reference goes away and normal refcounting
    # (+ Dask's client-side GC of non-fire_and_forget futures) can reclaim
    # them. Non-done futures are kept so the running task isn't cancelled.
    self._tracked_futures = [f for f in self._tracked_futures if not f.done()]


VARIANTS = {
    "baseline": submit_batch_baseline,
    "done_callback_release": submit_batch_done_callback_release,
    "tracked_list_prune": submit_batch_tracked_list_prune,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--n-tasks", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=5,
                         help="record an RSS sample every N batches")
    args = parser.parse_args()

    # Monkeypatch the variant under test onto the class before any instance
    # is created, so DaskExecutor's own submit_batch body is swapped out
    # cleanly for exactly this run.
    DaskExecutor.submit_batch = VARIANTS[args.variant]

    run_dir_base = os.path.join(HERE, f"run_dirs_{args.variant}")
    if os.path.isdir(run_dir_base):
        shutil.rmtree(run_dir_base)
    os.makedirs(run_dir_base, exist_ok=True)

    # DaskLocalLogPlugin (registered by start_cluster) reads log_dir from
    # the LoggerConfig singleton, normally populated by run.py's main() via
    # setup_logger -- bypassed here since this harness drives DaskExecutor
    # directly, so it's set up explicitly instead.
    setup_logger(run_dir_base, "INFO", log_dir="logs")

    executor = DaskExecutor(
        LocalCluster_config={"n_workers": args.n_workers, "threads_per_worker": 1},
    )
    executor.start_cluster()

    runner_config = {
        "type": "ExampleRunner",
        "__runner_name": "example",
        "sleep_sec": 0.001,
    }

    proc = psutil.Process(os.getpid())
    rows = []
    t0 = time.time()

    n_batches = (args.n_tasks + args.batch_size - 1) // args.batch_size
    submitted = 0
    for b in range(n_batches):
        this_batch = min(args.batch_size, args.n_tasks - submitted)
        pairs = []
        for j in range(this_batch):
            idx = submitted + j
            run_dir = os.path.join(run_dir_base, f"task_{idx}")
            pairs.append((run_dir, {"c1": float(idx), "c2": 1.0}))

        executor.submit_batch(pairs, runner_config, client=executor.client)
        submitted += this_batch

        if b % args.sample_every == 0 or b == n_batches - 1:
            gc.collect()
            rss_mb = proc.memory_info().rss / 1e6
            elapsed = time.time() - t0
            rows.append((submitted, elapsed, rss_mb))
            print(f"[{args.variant}] submitted={submitted}/{args.n_tasks} "
                  f"elapsed={elapsed:.1f}s rss={rss_mb:.1f}MB", flush=True)

    # Wait for all submitted tasks to actually finish (not just be
    # submitted) before the final measurement, by polling the scheduler's
    # own task-state counts rather than a fixed sleep -- submission
    # (microseconds/task) vastly outpaces completion (ExampleRunner's own
    # sleep_sec + subprocess overhead), so a short fixed sleep would measure
    # RSS while most tasks are still queued/running, understating any
    # completed-but-never-released leak.
    drain_t0 = time.time()
    drain_timeout = max(60.0, args.n_tasks * 0.05)
    while time.time() - drain_t0 < drain_timeout:
        info = executor.client.scheduler_info()
        processing = sum(len(w.get("processing", {})) for w in info.get("workers", {}).values())
        # Also check the scheduler's own task count via a lighter-weight
        # proxy: nothing processing AND client reports no pending futures
        # for the LAST batch submitted (best-effort; ExampleRunner's own
        # output.txt writes are the real completion signal, but polling
        # scheduler_info avoids needing to track every individual future).
        if processing == 0:
            break
        time.sleep(0.5)
    else:
        print(f"[{args.variant}] WARNING: drain timeout ({drain_timeout:.0f}s) reached "
              f"with tasks still processing", flush=True)

    time.sleep(2)  # short settle after processing hits 0
    gc.collect()
    rss_mb = proc.memory_info().rss / 1e6
    elapsed = time.time() - t0
    rows.append((submitted, elapsed, rss_mb))
    print(f"[{args.variant}] FINAL submitted={submitted} elapsed={elapsed:.1f}s rss={rss_mb:.1f}MB", flush=True)

    out_path = os.path.join(RESULTS_DIR, f"{args.variant}.csv")
    with open(out_path, "w") as f:
        f.write("n_submitted,elapsed_s,rss_mb\n")
        for n, e, r in rows:
            f.write(f"{n},{e:.2f},{r:.2f}\n")
    print(f"Wrote {out_path}")

    executor.client.close()
    executor.cluster.close()
    shutil.rmtree(run_dir_base, ignore_errors=True)


if __name__ == "__main__":
    main()
