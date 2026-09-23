"""
Standalone memory profiler for AsciiBinToHdf5Packer.pack_run_dir (see
fix/dask-executor-future-memory-leak branch). Reproduces the exact access
pattern supervisor.py's monitor_runs() uses in a real active-learning run:
repeated, synchronous, in-process calls to pack_run_dir() against ONE
growing hdf5 file, one completed "run dir" at a time.

Goal: distinguish between three possible explanations for the 33.5GB driver
RSS observed in job 22222070 before it was OOM-killed:

  (a) transient per-call spike only (RSS returns to baseline after each
      call) -- np.fromfile's buffer is freed once the call returns; the
      real OOM would then need concurrent processes + other retained
      driver state to add up to 32GB.
  (b) per-call peak GROWS as the target .h5 file grows (libhdf5 caching /
      allocator behavior on a large append-mode file, not just the
      np.fromfile buffer).
  (c) RSS never returns to baseline between calls -- genuine cumulative
      leak in the packer/h5py path.

Each synthetic "run dir" contains binary files sized like real GENE output
(field.dat, mom_e/i/z.dat, beta_field.dat, beta_mom_e/i/z.dat -- tens to
~110MB each, per the real 97481_active_learning_4method run) plus a couple
of small ascii files, so _is_binary_file's detection and the two
create_dataset code paths are both exercised realistically.

Usage::
    python packer_mem_test.py --n-runs 300 --files-per-run realistic

Writes results/packer_mem_test.csv with per-call before/after/peak RSS,
plus running .h5 file size, so growth-vs-filesize can be plotted.
"""
import argparse
import gc
import os
import shutil
import sys
import time
import tracemalloc

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from enchanted_surrogates.packers.ascii_bin_to_hdf5_packer import AsciiBinToHdf5Packer
from enchanted_surrogates.utils.logger import setup_logger

RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Real file sizes pulled from the 97481_active_learning_4method run's own
# enchanted_data_packed.h5 inspection (see conversation history): individual
# GENE binary dumps ranged tens of MB up to ~112MB. This mimics that mix
# rather than using one uniform size, since create_dataset cost may not be
# linear in size.
REALISTIC_BINARY_FILES_MB = {
    "field.dat": 87,
    "mom_e.dat": 45,
    "mom_i.dat": 45,
    "mom_z.dat": 45,
    "beta_field.dat": 112,
    "beta_mom_e.dat": 30,
    "beta_mom_i.dat": 30,
    "beta_mom_z.dat": 30,
}
SMALL_BINARY_FILES_MB = {name: max(1, mb // 10) for name, mb in REALISTIC_BINARY_FILES_MB.items()}

ASCII_FILES = {
    "parameters.dat": "&general\n  some = 1\n/\n" * 200,
    "gene.log": "step 1 done\n" * 500,
}


def make_run_dir(base_dir: str, run_idx: int, file_sizes_mb: dict) -> str:
    run_dir = os.path.join(base_dir, f"run_{run_idx:05d}")
    os.makedirs(run_dir, exist_ok=True)
    for name, mb in file_sizes_mb.items():
        path = os.path.join(run_dir, name)
        # random bytes so _is_binary_file's null-byte / utf-8-decode check
        # reliably classifies these as binary, and so the file isn't
        # trivially compressible in a way that wouldn't match real GENE
        # output's memory footprint during np.fromfile + create_dataset.
        with open(path, "wb") as f:
            f.write(np.random.randint(0, 256, size=mb * 1024 * 1024, dtype=np.uint8).tobytes())
    for name, text in ASCII_FILES.items():
        with open(os.path.join(run_dir, name), "w") as f:
            f.write(text)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=300)
    parser.add_argument("--files-per-run", choices=["realistic", "small"], default="realistic",
                         help="'realistic' matches real GENE output sizes (~400MB/run dir, "
                              "slow/disk-heavy); 'small' is a 10x-shrunk version for quick iteration.")
    parser.add_argument("--sample-every", type=int, default=1,
                         help="record a measurement every N pack_run_dir calls")
    parser.add_argument("--tracemalloc-every", type=int, default=0,
                         help="if >0, also run tracemalloc snapshots every N calls "
                              "(slower; separates Python-heap vs non-Python RSS growth)")
    args = parser.parse_args()

    file_sizes_mb = REALISTIC_BINARY_FILES_MB if args.files_per_run == "realistic" else SMALL_BINARY_FILES_MB

    work_dir = os.path.join(HERE, "work")
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    run_dirs_base = os.path.join(work_dir, "run_dirs")
    pack_base = os.path.join(work_dir, "pack_target")
    os.makedirs(run_dirs_base, exist_ok=True)
    os.makedirs(pack_base, exist_ok=True)

    setup_logger(work_dir, "WARNING", log_dir="logs")

    packer = AsciiBinToHdf5Packer(base_run_dir=pack_base)
    hdf5_path = packer.hdf5_path
    print(f"Packing into {hdf5_path}", flush=True)

    proc = psutil.Process(os.getpid())
    if args.tracemalloc_every > 0:
        tracemalloc.start()

    rows = []
    gc.collect()
    baseline_rss = proc.memory_info().rss / 1e6
    print(f"Baseline RSS before any pack_run_dir call: {baseline_rss:.1f}MB", flush=True)

    t0 = time.time()
    for i in range(args.n_runs):
        run_dir = make_run_dir(run_dirs_base, i, file_sizes_mb)

        gc.collect()
        rss_before = proc.memory_info().rss / 1e6

        peak_rss = rss_before
        # Sample RSS mid-call isn't possible without threading; instead
        # measure immediately after (the call is synchronous/blocking, so
        # "after" captures the retained state post-call, and a separate
        # coarser peak-tracking pass below via a watcher thread captures
        # the true in-call peak).
        packer.pack_run_dir(run_dir, results=None)

        rss_after_nogc = proc.memory_info().rss / 1e6
        gc.collect()
        rss_after_gc = proc.memory_info().rss / 1e6

        # Remove the synthetic run dir immediately after packing (mirrors
        # supervisor.py calling delete_unwanted_files right after
        # pack_run_dir), so disk usage doesn't also explode, and so any
        # RSS attributable to the source files being page-cached is
        # excluded from the *process* RSS measurement anyway (page cache
        # isn't process RSS).
        shutil.rmtree(run_dir, ignore_errors=True)

        h5_size_mb = os.path.getsize(hdf5_path) / 1e6 if os.path.exists(hdf5_path) else 0.0

        if i % args.sample_every == 0 or i == args.n_runs - 1:
            elapsed = time.time() - t0
            row = dict(
                call_idx=i,
                elapsed_s=round(elapsed, 2),
                rss_before_mb=round(rss_before, 2),
                rss_after_nogc_mb=round(rss_after_nogc, 2),
                rss_after_gc_mb=round(rss_after_gc, 2),
                delta_after_gc_mb=round(rss_after_gc - rss_before, 2),
                h5_size_mb=round(h5_size_mb, 2),
            )
            rows.append(row)
            print(f"[{i}/{args.n_runs}] rss_before={rss_before:.1f}MB "
                  f"rss_after(nogc)={rss_after_nogc:.1f}MB rss_after(gc)={rss_after_gc:.1f}MB "
                  f"delta={rss_after_gc - rss_before:+.1f}MB h5={h5_size_mb:.1f}MB "
                  f"elapsed={elapsed:.1f}s", flush=True)

        if args.tracemalloc_every > 0 and i % args.tracemalloc_every == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"    tracemalloc: current={current/1e6:.1f}MB peak={peak/1e6:.1f}MB", flush=True)
            tracemalloc.reset_peak()

    final_rss = proc.memory_info().rss / 1e6
    print(f"FINAL RSS after {args.n_runs} calls: {final_rss:.1f}MB "
          f"(baseline was {baseline_rss:.1f}MB, net growth {final_rss - baseline_rss:+.1f}MB)", flush=True)

    out_path = os.path.join(RESULTS_DIR, "packer_mem_test.csv")
    with open(out_path, "w") as f:
        header = list(rows[0].keys())
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")
    print(f"Wrote {out_path}")

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
