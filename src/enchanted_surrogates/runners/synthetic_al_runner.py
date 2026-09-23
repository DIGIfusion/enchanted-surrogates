import os
import time

import numpy as np

from enchanted_surrogates.runners.base_runner import Runner

CLASSES = ["ETG", "ITG", "KBM", "MTM", "TEM"]


class SyntheticAlRunner(Runner):
    """
    Fast stand-in for the real HELENA/GENE runners, used only by
    packer_memory_test/supervisor_mem_test.py to exercise the real
    Supervisor/DaskExecutor/sampler pipeline end-to-end without running
    actual simulations. Produces exactly the columns
    SvmTimeAwareActiveSampler / ExtraTreesTimeAwareActiveSampler need
    (a categorical class label + a continuous time value) plus a
    realistic-size ascii+binary file set in run_dir so the packer's
    pack_run_dir does real (if synthetic) I/O, matching the real run's
    driver-side workload as closely as possible while finishing in
    milliseconds instead of minutes.
    """

    def __init__(self, *args, **kwargs):
        self.sleep_sec = kwargs.get("sleep_sec", 0.001)
        self.write_files = bool(kwargs.get("write_files", False))
        self.binary_file_sizes_mb = kwargs.get(
            "binary_file_sizes_mb",
            {"field.dat": 1, "mom_e.dat": 1},
        )

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        os.makedirs(run_dir, exist_ok=True)
        time.sleep(self.sleep_sec)

        rng = np.random.default_rng()
        result = {
            "success": True,
            "helena_dir": run_dir,
            "gene_daniel_tree_classifier": str(rng.choice(CLASSES)),
            "runtime_sec_gene": float(rng.uniform(100, 300)),
            "runtime_sec_helena": float(rng.uniform(50, 150)),
        }

        if self.write_files:
            for name, mb in self.binary_file_sizes_mb.items():
                path = os.path.join(run_dir, name)
                with open(path, "wb") as f:
                    f.write(rng.integers(0, 256, size=mb * 1024 * 1024, dtype=np.uint8).tobytes())
            with open(os.path.join(run_dir, "gene.log"), "w") as f:
                f.write("step 1 done\n" * 200)

        return result
