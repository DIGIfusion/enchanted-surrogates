from .base_runner import Runner

class DoNothingRunner(Runner):
    """
    """    

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        """
        output = {'success':True}
        return output
