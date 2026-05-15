import os
from .base_runner import Runner

from enchanted_surrogates.utils.logger import get_logger
log = get_logger(__name__)

class SumRunner(Runner):
    """
    Example runner for sequential (or nested) workflows. Sums parameters together.
    
    Allows different way of operation based on parameters given in config files, which makes
    this useful for workflow testing. In real use, it would make more sense to implement 
    separate runners for different purposes.

    Example use:
    - Samples for three parameters, `c1`, `c2` and `c3` are generated
    - First SumRunner is defined with `params_to_sum: ['c1', 'c2']`
        - Result of `single_code_run` will be `c1 + c2`
    - Second SumRunner is defined with `params_to_sum: ['output', 'c3']`
        - Result of `single_code_run` will be `result_of_first_runner + c3`
    """

    def __init__(self, params_to_sum: list[str], **kwargs):
        """
        Args:
            params_to_sum (list[str]): List of parameter names
            fail_if (int, optional): Runner fails if calculated sum equals this value. 
                None by default
        """
        self.params_to_sum = params_to_sum
        self.fail_if = kwargs.get("fail_if", None)

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        Sums all params together (if they are in the list given to init function).
        Also writes all the parameters received as input into a text file.

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
        success = True
        for key, value in params.items():
            if key in self.params_to_sum:
                output += value

        epsilon = 0.25
        if self.fail_if and abs(output - self.fail_if) < epsilon:
            success = False

        return {
            "output": output,
            "success": success
        }
