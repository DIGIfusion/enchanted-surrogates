"""
Focused cProfile repro for the SvmTimeAwareActiveSampler hang observed in
supervisor_mem_test_config.yaml: get_next_samples() stalled for 9+ minutes
of CPU-bound (state R) time going from batch 6 to batch 7 (120 submitted
samples), with no new run dirs being created in the meantime -- i.e. the
stall is inside the sampler's own batch-selection code, not simulation
execution. This script drives the sampler directly (no Dask/executor/
supervisor machinery) through the same number of batches with fake results,
then profiles the next get_next_samples() call to find exactly which
function is eating the time.
"""
import cProfile
import io
import os
import pstats
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from enchanted_surrogates.samplers.svm_time_aware_active_sampler import SvmTimeAwareActiveSampler
from enchanted_surrogates.utils.logger import setup_logger

CLASSES = ["ETG", "ITG", "KBM", "MTM", "TEM"]

PARAMETERS = [
    'kymin', 'x0_norm_ped_width', 'T_eped', 'n_eped', 'd_n_ped',
    'd_T_ped', 'n_esep', 'Ti_ov_Te_ped', 'Ti_ov_Te_sep', 'd_Ti_ped_ov_d_Te_ped',
]
BOUNDS = [
    [0.05, 1], [0.05, 0.95], [0.467, 0.667], [7.95, 9.95], [0.016, 0.046],
    [0.047, 0.087], [3.61, 5.61], [0.82, 1.12], [3.11, 3.31], [0.52, 0.72],
]

work_dir = os.path.join(HERE, "work_profile")
os.makedirs(work_dir, exist_ok=True)
setup_logger(work_dir, "WARNING", log_dir="logs")

sampler = SvmTimeAwareActiveSampler(
    acquisition_mode="margin",
    parameters=PARAMETERS,
    bounds=BOUNDS,
    class_output_variable="gene_daniel_tree_classifier",
    time_output_variable="runtime_sec_gene",
    base_run_dir=work_dir,
    total_pool_size=200000,
    pool_chunk_size=5000,
    batch_size=20,
    initial_batch_size=20,
    budget=2000,
    exploration_per_batch=5,
    svc_kwargs=dict(kernel="rbf", C=1000.0, gamma="scale", class_weight=None),
    max_predicted_time=1800,
    cpuh_budget=999999,
    gene_cores_per_run=128,
    helena_cores_per_run=1,
    write_batch_info_every=20,
    seed=0,
)

rng = np.random.default_rng(0)


def fake_future_df(samples_dict):
    n = len(samples_dict)
    df = pd.DataFrame(samples_dict)
    df["success"] = True
    df["gene_daniel_tree_classifier"] = rng.choice(CLASSES, size=n)
    df["runtime_sec_gene"] = rng.uniform(100, 300, size=n)
    df["runtime_sec_helena"] = rng.uniform(50, 150, size=n)
    return df


import psutil
proc = psutil.Process(os.getpid())

N_WARMUP = int(os.environ.get("N_WARMUP", "2"))
print(f"Running batches 0-{N_WARMUP-1} as warmup...", flush=True)
for b in range(N_WARMUP):
    t0 = time.time()
    samples = sampler.get_next_samples()
    t1 = time.time()
    df = fake_future_df(samples)
    sampler.register_future(df)
    t2 = time.time()
    rss_mb = proc.memory_info().rss / 1e6
    print(f"batch {b}: get_next_samples={t1-t0:.2f}s register_future={t2-t1:.2f}s "
          f"submitted={sampler.submitted} rss={rss_mb:.1f}MB", flush=True)

print(f"\nProfiling batch {N_WARMUP}...", flush=True)
profiler = cProfile.Profile()
profiler.enable()
t0 = time.time()
samples = sampler.get_next_samples()
t1 = time.time()
profiler.disable()
rss_mb = proc.memory_info().rss / 1e6
print(f"batch {N_WARMUP} get_next_samples took {t1-t0:.2f}s rss={rss_mb:.1f}MB", flush=True)

stream = io.StringIO()
stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
stats.print_stats(30)
print(stream.getvalue())

out_path = os.path.join(HERE, "results", "profile_batch6.txt")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write(stream.getvalue())
print(f"Wrote {out_path}")
