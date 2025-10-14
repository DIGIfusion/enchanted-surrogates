from .base_runner import Runner

class DoNothingRunner(Runner):
    """
    This runner can be useful when the sampler already has the output because it was simply read from an existing dataset.
    This runner alows such a sampler to be used in the current enchanted surrogates workflow with minimal modifications.
    """

    def __init__(self, *args, **kwargs):
        NotImplemented

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        """
        result = params
        result.update({'success':True})
        return result
