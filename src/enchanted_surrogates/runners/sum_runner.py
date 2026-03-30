import os
from .base_runner import Runner

from enchanted_surrogates.utils.logger import get_logger
log = get_logger(__name__)

class SumRunner(Runner):
    def __init__(self, params_to_sum: list[str], **kwargs):
        self.params_to_sum = params_to_sum

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        Sums all params together (if they are in the list given to init function).

        Args:
            run_dir (str): Run directory of this instance
            params (dict): All params from samplers

        Returns:
            out (dict): At minimum contains:
            - "output" (float): primary numeric result.
            - "success" (bool): True if the run produced an output.
        """
        if "run_dir" in params:
            log.info("Starting higher-order runner. Previous run dir was: " + params["run_dir"])
        else:
            log.info("Starting first-order runner")

        with open(os.path.join(run_dir, "params.txt"), "w") as out_file:
            out_file.write(str(params))

        output: float = 0.0
        for key, value in params.items():
            if key in self.params_to_sum:
                output += value

        return {
            "output": output,
            "success": True
        }
