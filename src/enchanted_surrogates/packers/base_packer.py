from abc import ABC, abstractmethod


class Packer(ABC):
    """
    Base class for packers that archive the contents of a run directory into
    a single storage file (e.g. HDF5), and can later unpack a previously
    archived run back out to disk.

    Subclasses implement `pack_run_dir` (and, where it makes sense, an
    `unpack_run_dir` counterpart) for a specific on-disk file layout and a
    specific storage format.
    """

    def __init__(self, **kwargs):
        pass

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
