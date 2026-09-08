from .base_packer import Packer

from enchanted_surrogates.utils.logger import get_logger
log = get_logger(__name__)


class MultiPacker(Packer):
    """
    Packer that fans a single run directory out to multiple other packers.

    Useful when a run's results should end up in more than one place at
    once, e.g. raw files archived to hdf5 via `AsciiBinToHdf5Packer` and
    parsed results appended to a csv via `CsvPacker`, without having to
    list multiple packers explicitly in the run group's `packer` sequence.
    """

    def __init__(self, **kwargs):
        """
        Args:
            packers (list[dict]): Configs for the packers to run, each a
                dict with a `type` key (the packer's class name, in
                snake_case or PascalCase) plus whatever kwargs that
                packer's constructor takes. Required.
            base_run_dir (str, optional): Forwarded to each sub-packer's
                constructor (as with any other packer), unless a sub-packer
                config already sets its own `base_run_dir`.
        """
        super().__init__(**kwargs)

        from enchanted_surrogates.utils.precise_imports import import_packer

        packer_configs = kwargs.get('packers')
        if not packer_configs:
            raise ValueError("MultiPacker requires a non-empty 'packers' list of packer configs")

        self.packers = []
        for packer_config in packer_configs:
            packer_settings = dict(packer_config)
            packer_type = packer_settings.pop('type')
            if self.base_run_dir is not None:
                packer_settings.setdefault('base_run_dir', self.base_run_dir)
            self.packers.append(import_packer(packer_type, packer_settings))

    def pack_run_dir(self, run_dir: str, results: dict = None) -> dict:
        """
        Run every configured sub-packer's `pack_run_dir` on `run_dir`.

        Each sub-packer is run in turn; if one raises, the error is logged
        and the remaining sub-packers still run, so one misbehaving packer
        does not prevent the others from packing this run.

        Args:
            run_dir (str): Path to the run directory to pack.
            results (dict, optional): Results previously parsed from
                `run_dir`, forwarded unchanged to each sub-packer.

        Returns:
            dict: Mapping from each sub-packer's class name to what it
            returned from `pack_run_dir`, or the exception it raised.
        """
        outcomes = {}
        for packer in self.packers:
            name = packer.__class__.__name__
            try:
                outcomes[name] = packer.pack_run_dir(run_dir, results)
            except Exception as exc:
                log.error("Sub-packer '%s' failed on run dir '%s'", name, run_dir, exc_info=True)
                outcomes[name] = exc
        return outcomes
