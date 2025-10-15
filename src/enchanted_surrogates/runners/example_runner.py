import os
import sys
import numbers
from collections.abc import Iterable
from datetime import datetime
from time import sleep
import numpy as np

from .base_runner import Runner
from enchanted_surrogates.utils.is_package_available import is_package_available

if is_package_available('dask'):
    from dask.distributed import print


def is_number(x):
    return isinstance(x, numbers.Number)


def is_iterable(x, *, treat_strings_as_iterable=True):
    if not isinstance(x, Iterable):
        return False
    if not treat_strings_as_iterable and isinstance(x, (str, bytes, bytearray)):
        return False
    return True


class ExampleRunner(Runner):
    """
    ExampleRunner: a minimal Runner implementation that executes a tiny example workload,
    reads a numeric output from a file, and returns a simple result dictionary.

    Overview
    - Performs a single short Python one-liner that sums two parameters and appends the
      numeric result to "<run_dir>/output.txt".
    - Supports three parameter modes (0, 1, 2); each mode expects two parameter keys
      and returns a single primary numeric output under "output_1", "output_2", or
      "output_3" respectively, together with a boolean "success" flag and optional
      diagnostic strings.
    - Intended as a lightweight example and template for implementing real runners.

    Initialization parameters (via kwargs)
    - parameter_mode (int, default 0): selects which pair of parameter names the run
      will read from `params` and which output key will be used.
      - 0 -> expects params["c1"], params["c2"] -> returns "output_1"
      - 1 -> expects params["c3"], params["c4"] -> returns "output_2"
      - 2 -> expects params["c5"], params["c6"] -> returns "output_3"
    - sleep_sec (number or two-element iterable, default 0.01): controls the pause
      after executing the example command. See Sleep sec behavior below.

    Sleep sec behavior
    - Fixed sleep: pass a number (int, float, numpy scalar). The runner will sleep
      that many seconds after execution.
    - Random sleep: pass a two-element iterable [low, high] (list, tuple, numpy array).
      Each run will draw a single sample from a uniform distribution on [low, high]
      and sleep that many seconds.
    - Validation: the implementation expects exactly two bounds for random sleep; if
      a provided iterable has length not equal to two the runner should raise ValueError.
      Ensure bounds are numeric and optionally enforce low <= high.
    - Usage: call sleep(self.get_sleep_sec()) so the numeric duration returned by
      get_sleep_sec is passed to time.sleep.

    single_code_run behavior and return contract
    - Writes to and reads from "<run_dir>/output.txt"; repeated runs append unless
      the file is cleared by the caller.
    - Converts the file contents to float and places that value in the primary output
      key. Non-numeric or multi-line contents will raise an exception.
    - Returned dictionary values should be non-iterable base types (int, float, str, bool)
      so they integrate cleanly with dataset creation and surrogate tooling.

    Errors and edge cases
    - Raises ValueError for unsupported parameter_mode.
    - Raises ValueError for invalid random sleep bounds (not exactly two values or low > high).
    - Raises TypeError if sleep_sec is neither a number nor a two-element iterable.

    Example usage
    - runner = ExampleRunner(parameter_mode=0, sleep_sec=(0.1, 0.5))
    - runner.single_code_run("/tmp/run1", params={"c1": 1.2, "c2": 2.3})
      -> {"output_1": 3.5, "success": True, ...}
    """

    def __init__(self, *args, **kwargs):
        self.parameter_mode = kwargs.get('parameter_mode', 0)
        self.sleep_sec = kwargs.get('sleep_sec', 0.01)
    
    def get_sleep_sec(self):
        """
        Return a numeric sleep duration (seconds) derived from self.sleep_sec.

        Behavior
        - If self.sleep_sec is a numeric value (int, float, numpy scalar), return that
        value cast to float.
        - If self.sleep_sec is an iterable of exactly two numeric values, interpret them
        as lower and upper bounds and return a single sample drawn from the uniform
        distribution on [low, high].
        - If self.sleep_sec is an iterable with length not equal to two, raise ValueError.
        - If self.sleep_sec is neither a number nor a valid two-element iterable, raise
        TypeError.

        Parameters
        - self.sleep_sec: either a number (fixed sleep seconds) or a two-element iterable
        [low, high] specifying uniform random bounds.

        Returns
        - float: sleep duration in seconds suitable for time.sleep.

        Raises
        - ValueError: when a two-element iterable is required but self.sleep_sec has a
        different length or when low > high.
        - TypeError: when self.sleep_sec is not a number and not an iterable of two numbers.

        Examples
        - self.sleep_sec = 0.2   -> returns 0.2
        - self.sleep_sec = (0.1, 0.5) -> returns a float uniformly sampled between 0.1 and 0.5
        """
        if is_number(self.sleep_sec):
            return self.sleep_sec
        elif is_iterable(self.sleep_sec):
            if len(self.sleep_sec) > 2:
                raise ValueError('RANDOM SLEEP BOUNDS MUST BE AN ITERABLE WITH ONLY TWO VALUES, LOWER BOUND AND UPPER BOUND. SELF.SLEEP_SEC HAS MORE THAN TWO VALUES')
            return np.random.uniform(*self.sleep_sec)

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        Execute a single example run in `run_dir` using the current parameter_mode and
        return a simple results dictionary suitable for dataset creation and active
        learning workflows.

        Behavior
        - Writes the numeric result of a short Python one-liner to "<run_dir>/output.txt".
        - Sleeps for a duration determined by `self.get_sleep_sec()` after execution.
        - Reads the numeric output back from the file, converts it to float, and returns
        a dict containing a primary output key, a boolean "success" flag, and
        optional diagnostic strings.

        Parameter modes
        - parameter_mode == 0: expects params["c1"], params["c2"]; result returned under "output_1".
        - parameter_mode == 1: expects params["c3"], params["c4"]; result returned under "output_2".
        - parameter_mode == 2: expects params["c5"], params["c6"]; result returned under "output_3".
        - any other parameter_mode raises ValueError.

        Parameters
        - run_dir (str): Directory path where "output.txt" will be created and read.
        - params (dict, optional): Parameter mapping required by the chosen parameter_mode.

        Returns
        - dict: At minimum contains:
        - "output_n" (float): primary numeric result (output_1 / output_2 / output_3).
        - "success" (bool): True if the run produced an output.
        - Additional diagnostic keys may be included as non-iterable base types.

        Notes and constraints
        - The method appends to "output.txt"; repeated runs will append unless the caller
        clears the file beforehand.
        - The method converts the full contents of "output.txt" to float; non-numeric or
        multi-line contents will raise an exception during conversion.
        - Returned values should be non-iterable base types (int, float, str, bool) so they
        are compatible with dataset and surrogate tooling.
        """

        
        # Implementation for the example runner
        # Logic should follow something like
        # - via parser, write some input files there
        # - run the code
        # - return outputs
        #   - 'output' should be the most important output used in active learning methods,
        #     it should be a single float, integer or class
        #   - 'success' indicates if the run sucessfully created output or if it crashed etc
        #   - Any other code outputs of interest can be places in the returned dictionary. They
        #     will be collected and available in enchanted surrogates created datasets.
        #   - for the datasets to be created properly the return dictionary values should not be
        #     iterables, only base types such as int, float, string, boolean...

        outfile = os.path.join(run_dir, "output.txt")

        # TODO make example parser
        # parser.write_input(run_dir, params)
        if self.parameter_mode == 0:
            c1 = params["c1"]
            c2 = params["c2"]
            cmd = f'{sys.executable} -c "print({c1} + {c2})" >> "{outfile}"'
            os.system(cmd)
            with open(outfile, 'r') as f:
                output = float(f.read().strip())
            sleep_sec = self.get_sleep_sec()
            print(f'{datetime.now()} IN EXAMPLE RUNNER - SLEEPING FOR: {sleep_sec}')
            sleep(sleep_sec)
            result = {
                "output_1": output, "success": True, 'other_code_output_A': 'something_from_code_A'}
        elif self.parameter_mode == 1:
            c3 = params["c3"]
            c4 = params["c4"]
            cmd = f'{sys.executable} -c "print({c3} + {c4})" >> "{outfile}"'
            os.system(cmd)
            with open(outfile, 'r') as f:
                output = float(f.read().strip())
            sleep_sec = self.get_sleep_sec()
            print(f'{datetime.now()} IN EXAMPLE RUNNER - SLEEPING FOR: {sleep_sec}')
            sleep(sleep_sec)
            result = {
                "output_2": output, "success": True, 'other_code_output_B': 'something_from_code_B'}
        elif self.parameter_mode == 2:
            c5 = params["c5"]
            c6 = params["c6"]
            cmd = f'{sys.executable} -c "print({c5} + {c6})" >> "{outfile}"'
            os.system(cmd)
            with open(outfile, 'r') as f:
                output = float(f.read().strip())
            sleep_sec = self.get_sleep_sec()
            print(f'{datetime.now()} IN EXAMPLE RUNNER - SLEEPING FOR: {sleep_sec}')
            sleep(sleep_sec)
            result = {
                "output_3": output, "success": True, 'other_code_output_C': 'something_from_code_C'}
        else:
            raise ValueError(
                'THE SET PARAMETER MODE DOES NOT MATCH ANY KNOWN PARAMETER MODE FOR EXAMPLE RUNNER')

        # TODO execute some code actually
        # for now, just sum the two parameters

        # TODO read output
        # parser.read_output(run_dir)

        return result
