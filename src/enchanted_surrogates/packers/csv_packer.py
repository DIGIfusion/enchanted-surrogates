import csv
import io
import math
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

    Packing a run directory that is already in the csv file (a resumed ES run
    or a rerun of the run directory) never adds a second row: with
    `on_existing: skip` (default) the existing row is kept, with `overwrite` it
    is replaced, see `Packer`.

    The csv file is only fully read and rewritten when it has to be. The header
    and the set of packed run directories are read from the file once, on the
    first call, and kept in memory (so a resumed run sees what was packed
    before the interruption). A new run directory whose keys all already have a
    column is then appended as one line. Rewriting the whole file only happens
    for a row that adds a new column, or for `overwrite` of a run directory
    that is already packed. The cache assumes this packer is the only writer of
    the file while it runs (true when the main scheduler does the packing).
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

        # filled on first use by `_load_cache`, kept up to date by every write
        self._cache_loaded = False
        self._columns = None   # header of the csv file, None while it does not exist
        self._seen = set()     # run_dirs that have a row in the csv file

    def _csv_exists(self) -> bool:
        return os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

    def _drop_torn_last_line(self) -> None:
        """
        If the csv file does not end in a newline the last write was cut off
        (e.g. the job was killed mid-append). Truncate the file back to the last
        complete line, so the next append does not glue a row onto the broken
        one. The dropped run dir is not in the file, so it is packed again.
        """
        with open(self.csv_path, 'rb+') as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) == b'\n':
                return
            f.seek(0)
            content = f.read()
            keep = content.rfind(b'\n') + 1
            log.warning("%s ends in an incomplete line, dropping it", self.csv_path)
            f.truncate(keep)

    def _load_cache(self) -> None:
        """Read the header and the packed run_dirs from the csv file (once)."""
        if self._csv_exists():
            self._drop_torn_last_line()
        if self._csv_exists():
            columns = list(pd.read_csv(self.csv_path, nrows=0).columns)
            self._columns = columns
            if 'run_dir' in columns:
                self._seen = set(pd.read_csv(self.csv_path, usecols=['run_dir'])['run_dir'])
        self._cache_loaded = True

    @staticmethod
    def _csv_value(value):
        """A value as it is written by `DataFrame.to_csv`: numpy scalars as python ones, NaN/None blank."""
        if hasattr(value, 'item') and getattr(value, 'ndim', None) == 0:
            value = value.item()
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ''
        return value

    def _append_row(self, row: dict) -> None:
        """Append `row` as one line, aligned to the existing header. Every key of `row` must be a column."""
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=self._columns, restval='', lineterminator='\n')
        writer.writerow({k: self._csv_value(v) for k, v in row.items()})
        # a single write call, so a line is either all in the file or cut off at the end
        with open(self.csv_path, 'a', newline='') as f:
            f.write(buffer.getvalue())
        self._seen.add(row['run_dir'])

    def _write_atomic(self, df: pd.DataFrame) -> None:
        """Replace the csv file by `df` via a temporary file, so a kill never leaves a half-written file."""
        tmp_path = self.csv_path + '.tmp'
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.csv_path)
        self._columns = list(df.columns)
        self._seen = set(df['run_dir'])

    def pack_run_dir(self, run_dir: str, results: dict = None) -> None:
        """
        Write `results` as a single row to this packer's csv file: appended
        (one line, without rereading the file, when every key already has a
        column), or, if the csv file already has a row for `run_dir`, kept as it is
        (`on_existing: skip`, default) or replaced (`on_existing: overwrite`).

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
                if not self._cache_loaded:
                    self._load_cache()
                already_packed = run_dir in self._seen
                if already_packed and self.on_existing == 'skip':
                    return
                if not already_packed and self._columns is not None and all(k in self._columns for k in row):
                    self._append_row(row)
                    return
                # slow path: new column, first row, or overwrite of a packed run dir
                if self._csv_exists():
                    existing = pd.read_csv(self.csv_path)
                    same_run = existing['run_dir'] == run_dir if 'run_dir' in existing else pd.Series(False, index=existing.index)
                    if same_run.any():
                        # repacked run dir: overwrite the (first) row in place and drop any duplicates of it
                        first = same_run.idxmax()
                        for column in row:
                            if column not in existing:
                                existing[column] = pd.NA
                        existing = existing.astype(object)
                        existing.loc[first, :] = pd.Series(row).reindex(existing.columns)
                        combined = existing[~same_run | (existing.index == first)]
                    else:
                        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                else:
                    combined = pd.DataFrame([row])
                self._write_atomic(combined)
        except Exception:
            log.error("Failed to pack run dir '%s' into %s", run_dir, self.csv_path, exc_info=True)
