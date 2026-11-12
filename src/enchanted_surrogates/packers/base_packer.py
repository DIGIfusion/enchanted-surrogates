from abc import ABC, abstractmethod


class Packer(ABC):
    """
    Base class for packers that archive the contents of a run directory into
    a single storage file (e.g. HDF5), and can later unpack a previously
    archived run back out to disk.

    Subclasses implement `pack_run_dir` (and, where it makes sense, an
    `unpack_run_dir` counterpart) for a specific on-disk file layout and a
    specific storage format.

    Repacking is the packer's responsibility: `pack_run_dir` may be called for
    a run directory that is already packed. This happens when a crashed or
    interrupted ES run is resumed (`run_mode: resume`), because the supervisor
    sees the results of runs that finished before the interruption again, and
    when someone deliberately reruns a run directory. What the packer does is
    set by `on_existing` and must never be a duplicate:

    - `"skip"` (default): keep what is packed, do nothing. The supervisor
      deletes the files it does not keep after packing, so a run directory
      that is packed again after a resume is often almost empty, and
      overwriting would replace a complete packed copy with that.
    - `"overwrite"`: replace what is packed with the current content of the
      run directory. Use it when run directories are rerun and the packed data
      must be refreshed.
    """

    def __init__(self, **kwargs):
        # this will be assigned by the supervisor and can be used by packers to determine the base run directory for relative paths
        self.base_run_dir = kwargs.get("base_run_dir", None)
        # what to do when a run directory is already packed: "skip" (default) or "overwrite"
        self.on_existing = kwargs.get("on_existing", "skip")
        if self.on_existing not in ("skip", "overwrite"):
            raise ValueError(f"on_existing must be 'skip' or 'overwrite', not {self.on_existing!r}")

    @abstractmethod
    def pack_run_dir(self, run_dir: str, results: dict = None) -> dict:
        """
        Pack the contents of a single run directory into this packer's
        storage file.

        Args:
            run_dir (str): Path to the run directory to pack.
            results (dict, optional): Results previously parsed from
                `run_dir` by some other parser (e.g. a runner's output
                parser). Not every packer needs this - a packer that only
                archives raw files is free to ignore it - but it is passed
                through so packers that store parsed values alongside the
                raw files (rather than re-parsing them) can use it. May also
                be used to accumulate/return bookkeeping across repeated
                calls; see individual subclasses for how they use it.

        Returns:
            dict: Bookkeeping about what was packed. Subclasses define the
            exact contents.
        """
        raise NotImplementedError("Subclasses must implement this method")
