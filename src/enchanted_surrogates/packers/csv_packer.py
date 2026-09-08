import os
import threading

import pandas as pd

from .base_packer import Packer

from enchanted_surrogates.utils.logger import get_logger
log = get_logger(__name__)

_write_lock = threading.Lock()


def _default_csv_path() -> str:
    """
    Build the default csv output path in the user's home directory.

    The default file is `~/enchanted_packed.csv`. If that file already
    exists, `~/enchanted_packed_1.csv`, `_2.csv`, etc. are tried in turn
    until a path that does not yet exist is found.

    Returns:
        str: Path to the first available default csv file.
    """
    home = os.path.expanduser('~')
    base_name = 'enchanted_packed'
    path = os.path.join(home, f'{base_name}.csv')
    if not os.path.exists(path):
        return path

    n = 1
    while True:
        path = os.path.join(home, f'{base_name}_{n}.csv')
        if not os.path.exists(path):
            return path
        n += 1


class CsvPacker(Packer):
    """
    Packer that appends a run directory's parsed results to a single csv
    file, `enchanted_packed.csv` by default.

    Unlike packers that archive raw files, this packer only cares about the
    `results` dict passed to `pack_run_dir` (e.g. the row parsed from a
    run's `enchanted_datapoint.csv`). Each call appends one row to the csv
    file, adding a `run_dir` column so each row can be traced back to its
    source run directory.
    """

    def __init__(self, **kwargs):
        """
        Args:
            csv_path (str, optional): Path to the csv file to append to. If
                not given, defaults to `base_run_dir/enchanted_packed.csv`
                if `base_run_dir` is given, otherwise falls back to the
                first available path from `_default_csv_path`
                (`~/enchanted_packed.csv`, falling back to `_1.csv`,
                `_2.csv`, etc. if it already exists).
        """
        super().__init__(**kwargs)

        self.base_run_dir = kwargs.get('base_run_dir')
        if kwargs.get('csv_path') is not None:
            self.csv_path = kwargs['csv_path']
        elif self.base_run_dir is not None:
            os.makedirs(self.base_run_dir, exist_ok=True)
            self.csv_path = os.path.join(self.base_run_dir, 'enchanted_packed.csv')
        else:
            self.csv_path = _default_csv_path()

    def pack_run_dir(self, run_dir: str, results: dict = None) -> None:
        """
        Append `results` as a single row to this packer's csv file.

        The row includes a `run_dir` column (the absolute path to
        `run_dir`) alongside every key in `results`. If the csv file
        already exists, the new row's columns are aligned to the existing
        header, so new keys become new columns (filled with NaN for
        earlier rows) and missing keys are left blank for the new row.

        Errors are caught and logged rather than raised, so a single bad
        run does not interrupt a larger batch of calls.

        Args:
            run_dir (str): Path to the run directory that was just
                completed.
            results (dict, optional): Results parsed from `run_dir` by some
                other parser (e.g. a runner's output parser). If None, an
                empty dict is used, so the row only contains `run_dir`.

        Returns:
            None
        """
        run_dir = os.path.abspath(run_dir)
        row = {'run_dir': run_dir, **(results or {})}

        try:
            with _write_lock:
                if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
                    existing = pd.read_csv(self.csv_path)
                    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                    combined.to_csv(self.csv_path, index=False)
                else:
                    pd.DataFrame([row]).to_csv(self.csv_path, index=False)
        except Exception:
            log.error("Failed to pack run dir '%s' into %s", run_dir, self.csv_path, exc_info=True)
