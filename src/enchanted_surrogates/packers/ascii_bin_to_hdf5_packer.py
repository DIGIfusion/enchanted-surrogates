import os

import h5py
import numpy as np

from .base_packer import Packer

from enchanted_surrogates.utils.logger import get_logger
log = get_logger(__name__)

def _is_binary_file(path: str, chunk_size: int = 8192) -> bool:
    """
    Heuristically decide whether a file is binary or ascii text.

    Reads the first `chunk_size` bytes of `path` and treats the file as
    binary if it contains a null byte or the chunk cannot be decoded as
    utf-8.

    Args:
        path (str): Path to the file to inspect.
        chunk_size (int, optional): Number of bytes read from the start of
            the file to make the decision. Defaults to 8192.

    Returns:
        bool: True if the file looks binary, False if it looks like ascii
        text.
    """
    with open(path, 'rb') as f:
        chunk = f.read(chunk_size)
    if b'\x00' in chunk:
        return True
    try:
        chunk.decode('utf-8')
    except UnicodeDecodeError:
        return True
    return False


def _default_hdf5_path() -> str:
    """
    Build the default hdf5 output path in the user's home directory.

    The default file is `~/enchanted_data_packed.h5`. If that file already
    exists, `~/enchanted_data_packed_1.h5`, `_2.h5`, etc. are tried in turn
    until a path that does not yet exist is found.

    Returns:
        str: Path to the first available default hdf5 file.
    """
    home = os.path.expanduser('~')
    base_name = 'enchanted_data_packed'
    path = os.path.join(home, f'{base_name}.h5')
    if not os.path.exists(path):
        return path

    n = 1
    while True:
        path = os.path.join(home, f'{base_name}_{n}.h5')
        if not os.path.exists(path):
            return path
        n += 1


class AsciiBinToHdf5Packer(Packer):
    """
    Packer that archives a run directory's raw ascii and binary files into a
    single hdf5 file, preserving the run directory's tree structure, and can
    unpack a previously archived run back out to disk.

    All files under a run directory are assumed to be either ascii text or
    binary; each file is stored as-is (as text or as a raw byte array) rather
    than being parsed, so unpacking reproduces the original files exactly.

    Every run is stored under its own group named after the run directory's
    final path component, so `pack_run_dir` can be called repeatedly with
    different run directories and each is added alongside the previous ones
    in the same hdf5 file rather than overwriting it.
    """

    def __init__(self, **kwargs):
        """
        Args:
            hdf5_path (str, optional): Path to the hdf5 file to pack into /
                unpack from. If not given, defaults to the first available
                path from `_default_hdf5_path` (`~/enchanted_data_packed.h5`,
                falling back to `_1.h5`, `_2.h5`, etc. if it already exists).
        """
        self._custom_hdf5_path = False
        self._base_run_dir = None
        self._hdf5_path = None

        super().__init__(**kwargs)

        self.base_run_dir = kwargs.get('base_run_dir')
        if kwargs.get('hdf5_path') is not None:
            self.hdf5_path = kwargs['hdf5_path']
        elif self.base_run_dir is not None:
            self.hdf5_path = os.path.join(self.base_run_dir, 'enchanted_data_packed.h5')
        else:
            self.hdf5_path = _default_hdf5_path()

    @property
    def base_run_dir(self):
        return self._base_run_dir

    @base_run_dir.setter
    def base_run_dir(self, value):
        self._base_run_dir = value
        if value is not None and not self._custom_hdf5_path:
            os.makedirs(value, exist_ok=True)
            self._hdf5_path = os.path.join(value, 'enchanted_data_packed.h5')

    @property
    def hdf5_path(self):
        if self._hdf5_path is None:
            if self._base_run_dir is not None and not self._custom_hdf5_path:
                self._hdf5_path = os.path.join(self._base_run_dir, 'enchanted_data_packed.h5')
            elif not self._custom_hdf5_path:
                self._hdf5_path = _default_hdf5_path()
        return self._hdf5_path

    @hdf5_path.setter
    def hdf5_path(self, value):
        self._hdf5_path = value
        self._custom_hdf5_path = value is not None

    def _ensure_hdf5_path(self):
        """Move the default output into base_run_dir if that info becomes available later."""
        if self._custom_hdf5_path or self._base_run_dir is None:
            return

        self.base_run_dir = self._base_run_dir

    def pack_run_dir(self, run_dir: str, results: dict = None) -> None:
        """
        Pack all files under `run_dir` into this packer's hdf5 file.

        Walks `run_dir` recursively and stores each file as a dataset under
        `runs/<run_name>/...` in the hdf5 file, mirroring `run_dir`'s
        subdirectory structure as nested hdf5 groups. Each file is stored
        either as utf-8 text or as a raw `uint8` byte array, based on
        whether it is detected as ascii or binary (see `_is_binary_file`).
        If a run with the same name was already packed, it is deleted and
        replaced.

        Errors (per file, or packing `run_dir` as a whole) are caught and
        logged rather than raised, so a single bad run does not interrupt a
        larger batch of calls.

        Args:
            run_dir (str): Path to the run directory to pack. Only its final
                path component (e.g. "run_001") is used as the run's name in
                the hdf5 file.
            results (dict, optional): Results previously parsed from
                `run_dir` by some other parser. Accepted for interface
                compatibility with `Packer.pack_run_dir`, but ignored - this
                packer only archives raw files and has no use for parsed
                values.

        Returns:
            None
        """
        self._ensure_hdf5_path()
        run_dir = os.path.abspath(run_dir)
        run_name = os.path.basename(os.path.normpath(run_dir))

        try:
            with h5py.File(self.hdf5_path, 'a') as f:
                runs_group = f.require_group('runs')

                if run_name in runs_group:
                    log.warning(
                        "Run '%s' already exists in %s, overwriting", run_name, self.hdf5_path
                    )
                    del runs_group[run_name]
                run_group = runs_group.create_group(run_name)
                run_group.attrs['source_dir'] = run_dir

                for dirpath, _, filenames in os.walk(run_dir):
                    rel_dir = os.path.relpath(dirpath, run_dir)
                    group = (
                        run_group
                        if rel_dir == '.'
                        else run_group.require_group(rel_dir.replace(os.sep, '/'))
                    )

                    for filename in sorted(filenames):
                        file_path = os.path.join(dirpath, filename)

                        try:
                            if _is_binary_file(file_path):
                                data = np.fromfile(file_path, dtype=np.uint8)
                                dataset = group.create_dataset(filename, data=data)
                                dataset.attrs['file_type'] = 'binary'
                            else:
                                with open(file_path, 'r') as fh:
                                    text = fh.read()
                                dataset = group.create_dataset(
                                    filename, data=text, dtype=h5py.string_dtype(encoding='utf-8')
                                )
                                dataset.attrs['file_type'] = 'ascii'
                        except Exception:
                            log.error("Failed to pack file '%s'", file_path, exc_info=True)
        except Exception:
            log.error("Failed to pack run dir '%s' into %s", run_dir, self.hdf5_path, exc_info=True)

    def unpack_run_dir(self, run_dir: str, dest_dir: str) -> str:
        """
        Unpack a previously packed run back out to disk under `dest_dir`.

        Args:
            run_dir (str): Name of the run to unpack, as it was packed. Only
                the final path component is used, so either the run's
                original full path (e.g. "/path/to/run_001") or just its
                name (e.g. "run_001") work equally well.
            dest_dir (str): Directory to unpack the run into. The run is
                written to `dest_dir/<run_name>/`, reproducing the original
                subdirectory structure and file contents.

        Returns:
            str: Path to the unpacked run directory (`dest_dir/<run_name>`).

        Raises:
            KeyError: If no run named `run_name` exists in this packer's
                hdf5 file.
        """
        run_name = os.path.basename(os.path.normpath(run_dir))
        out_dir = os.path.join(dest_dir, run_name)

        with h5py.File(self.hdf5_path, 'r') as f:
            runs_group = f.get('runs')
            if runs_group is None or run_name not in runs_group:
                log.error("Run '%s' not found in %s", run_name, self.hdf5_path)
                raise KeyError(f"Run '{run_name}' not found in {self.hdf5_path}")

            self._unpack_group(runs_group[run_name], out_dir)

        log.debug("Unpacked run '%s' to %s", run_name, out_dir)
        return out_dir

    def _unpack_group(self, group: 'h5py.Group', out_dir: str) -> None:
        """
        Recursively write an hdf5 group's contents out to `out_dir`.

        Nested groups become subdirectories; datasets become files, written
        as text or as raw bytes based on their "file_type" attribute (see
        `pack_run_dir`).

        Args:
            group (h5py.Group): hdf5 group to unpack.
            out_dir (str): Directory to write this group's contents into.
                Created if it does not already exist.
        """
        os.makedirs(out_dir, exist_ok=True)

        for name, item in group.items():
            item_path = os.path.join(out_dir, name)

            if isinstance(item, h5py.Group):
                self._unpack_group(item, item_path)
                continue

            if item.attrs.get('file_type') == 'binary':
                np.asarray(item[()], dtype=np.uint8).tofile(item_path)
            else:
                text = item[()]
                if isinstance(text, bytes):
                    text = text.decode('utf-8')
                with open(item_path, 'w') as fh:
                    fh.write(text)
