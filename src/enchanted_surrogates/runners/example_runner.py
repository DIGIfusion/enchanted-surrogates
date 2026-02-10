import os
import sys
import subprocess
import numbers
from collections.abc import Iterable
from datetime import datetime
from time import sleep
import numpy as np

from .base_runner import Runner
from enchanted_surrogates.utils.logger import get_logger
from enchanted_surrogates.utils.is_package_available import is_package_available

log = get_logger(__name__)


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
    ExampleRunner: a minimal Runner implementation that executes a tiny example calculation,
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
    - sleep_sec (number or two-element iterable, default 0.01): controls the pause
      after executing the example command. See Sleep sec behavior below.
    - fail_prob (float in [0, 1], default 0): probability that the run raises a
      synthetic RuntimeError after successful execution. This is useful for testing
      failure handling in distributed schedulers and pipelines. The failure is raised
      only after writing and reading the output file and after sleeping for the
      computed sleep duration. When fail_prob == 0 no synthetic failures are triggered.

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
    - Raises ValueError for invalid random sleep bounds (not exactly two values or low > high).
    - Raises TypeError if sleep_sec is neither a number nor a two-element iterable.
    - If fail_prob is set in (0,1], a RuntimeError may be raised after a successful run
      to simulate stochastic failures; the probability of raising is equal to fail_prob.

    Example usage
    - runner = ExampleRunner(sleep_sec=(0.1, 0.5), fail_prob=0.1)
    - runner.single_code_run("/tmp/run1", params={"c1": 1.2, "c2": 2.3})
      -> {"output": 3.5, "success": True}
    """

    def __init__(self, *args, **kwargs):
        self.sleep_sec = kwargs.get("sleep_sec", 0.01)
        self.fail_prob = kwargs.get("fail_prob", 0)

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
            return float(self.sleep_sec)
        elif is_iterable(self.sleep_sec):
            # enforce exactly two-element iterable for random sleep bounds
            seq = list(self.sleep_sec)
            if len(seq) != 2:
                raise ValueError(
                    "RANDOM SLEEP BOUNDS MUST BE AN ITERABLE WITH EXACTLY TWO VALUES:"
                    " LOWER BOUND AND UPPER BOUND. SELF.SLEEP_SEC HAS LENGTH "
                    f"{len(seq)}"
                )
            low, high = seq
            if not (is_number(low) and is_number(high)):
                raise TypeError("RANDOM SLEEP BOUNDS MUST BE NUMERIC")
            if float(low) > float(high):
                raise ValueError("LOWER BOUND MUST BE <= UPPER BOUND FOR RANDOM SLEEP")
            return float(np.random.uniform(float(low), float(high)))
        else:
            raise TypeError("SELF.SLEEP_SEC MUST BE A NUMBER OR A TWO-ELEMENT ITERABLE")

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        """
        Execute a single example run in `run_dir` and
        return a simple results dictionary suitable for dataset creation and active
        learning workflows.

        Behavior
        - Writes the numeric result of a short Python one-liner to "<run_dir>/output.txt".
        - Sleeps for a duration determined by `self.get_sleep_sec()` after execution.
        - Reads the numeric output back from the file, converts it to float, and returns
          a dict containing a primary output key, a boolean "success" flag, and
          optional diagnostic strings.

        Parameters
        - run_dir (str): Directory path where "output.txt" will be created and read.
        - params (dict, optional): Parameter mapping.

        Returns
        - dict: At minimum contains:
          - "output" (float): primary numeric result.
          - "success" (bool): True if the run produced an output.

        Notes and constraints
        - The method appends to "output.txt"; repeated runs will append unless the caller
          clears the file beforehand.
        - The method converts the full contents of "output.txt" to float; non-numeric or
          multi-line contents will raise an exception during conversion.
        - Returned values should be non-iterable base types (int, float, str, bool) so they
          are compatible with dataset and surrogate tooling.
        - If fail_prob > 0, a RuntimeError may be raised after successful execution with
          probability equal to fail_prob to simulate stochastic failures for testing.
        """

        # Ensure run_dir exists
        os.makedirs(run_dir, exist_ok=True)

        if params is None:
            params = {}

        outfile = os.path.join(run_dir, "output.txt")

        # Parameter parsing: support 0, 1, or 2 provided parameters gracefully
        if len(params.keys()) == 0:
            c1 = 0.0
            c2 = 0.0
        elif len(params.keys()) == 1:
            c1 = params[list(params.keys())[0]]
            c2 = 0.0
        else:
            c1 = params[list(params.keys())[0]]
            c2 = params[list(params.keys())[1]]

        # Coerce numeric types to plain Python numbers for the one-liner to avoid weird print formats
        try:
            c1 = float(c1)
        except Exception:
            raise TypeError(f"Parameter c1 is not numeric: {c1!r}")
        try:
            c2 = float(c2)
        except Exception:
            raise TypeError(f"Parameter c2 is not numeric: {c2!r}")

        # Execute a tiny Python one-liner that prints the sum and append to outfile
        cmd = [sys.executable, "-c", f"print({c1} + {c2})"]

        with open(outfile, "a") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)

        if result.returncode != 0:
            raise RuntimeError(
                f"Command execution failed with exit code {result.returncode}"
            )

        # Read output and convert to float (strip whitespace)
        with open(outfile, "r") as f:
            contents = f.read().strip()
        try:
            output = float(contents)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse numeric output from {outfile}: {contents!r}"
            ) from exc

        # Sleep for configured duration
        sleep_sec = self.get_sleep_sec()
        log.info(f"IN EXAMPLE RUNNER - SLEEPING FOR: {sleep_sec}")
        sleep(sleep_sec)

        result = {"output": output, "success": True}

        # Synthetic failure injection for testing distributed failure handling
        if self.fail_prob is not None and self.fail_prob > 0:
            if not (is_number(self.fail_prob) and 0.0 <= float(self.fail_prob) <= 1.0):
                raise ValueError("fail_prob must be a number between 0 and 1")
            flip = np.random.uniform()
            if flip < float(self.fail_prob):
                raise RuntimeError(
                    f"THE RUN FAILED BECAUSE IT WAS UNLUCKY, fail_prob:{self.fail_prob}"
                )

        return result
