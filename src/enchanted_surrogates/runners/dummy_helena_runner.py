"""
DummyHelenaRunner copies the contents of an existing, already-completed HELENA
run directory into the new run_dir and returns immediately, instead of
actually running HELENA. Used to skip HELENA's ~9 minute runtime when
debugging something downstream (e.g. the GENE stage) that doesn't depend on
HELENA producing a *new* physics point.
"""
import os
import shutil
from datetime import datetime

from .base_runner import Runner
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)


class DummyHelenaRunner(Runner):
    """
    Initialization parameters (via kwargs):
      - source_helena_dir (str, required): path to an existing HELENA run_dir
        whose contents will be copied into every new run_dir.
    """

    def __init__(self, *args, **kwargs):
        self.source_helena_dir = kwargs["source_helena_dir"]

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        start_time = str(datetime.now())
        os.makedirs(run_dir, exist_ok=True)
        for name in os.listdir(self.source_helena_dir):
            src = os.path.join(self.source_helena_dir, name)
            dst = os.path.join(run_dir, name)
            if os.path.isfile(src):
                shutil.copy(src, dst)
        log.info(f"DummyHelenaRunner copied {self.source_helena_dir} -> {run_dir}")

        return {
            "success": True,
            "error": "",
            "start_time": start_time,
            "end_time": str(datetime.now()),
            "helena_dir": run_dir,
            "helena_error": "",
            "beta_n_achieved": None,
            "max_growthrate_mishka": None,
            "max_growthrate_castor": None,
        }
