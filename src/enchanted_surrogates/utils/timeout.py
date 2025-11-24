import concurrent.futures
import traceback
import logging
from typing import Any, Callable, Dict, Tuple

logger = logging.getLogger(__name__)

class FunctionTimeoutError(TimeoutError):
    """Raised when the wrapped function times out."""
    pass

class FunctionExecutionError(RuntimeError):
    """Raised when the wrapped function raises an exception in the worker process."""
    def __init__(self, message: str, worker_tb: str = None):
        super().__init__(message)
        # attach the worker traceback string (picklable)
        self.worker_tb = worker_tb

def run_with_timeout(func: Callable, timeout: float, /,
                     args: Tuple = (),
                     kwargs: Dict[str, Any] = None) -> Any:
    """
    Run func(*args, **(kwargs or {})) in a separate process and enforce timeout seconds.

    On success: returns the function's return value.
    On timeout: raises FunctionTimeoutError.
    If func raises: raises FunctionExecutionError with original traceback attached.

    Notes:
    - func and all arguments must be picklable.
    - This is cross-platform and will terminate the worker process on timeout.
    """
    if kwargs is None:
        kwargs = {}

    # Use a fresh ProcessPoolExecutor for each call so the worker is isolated
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as exe:
        fut = exe.submit(_worker_wrapper, func, args, kwargs)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            # Attempt to cancel and then raise our timeout error
            fut.cancel()
            raise FunctionTimeoutError(f"Function timed out after {timeout} seconds") from exc
        except Exception as exc:
            # If the worker raised FunctionExecutionError, it will be re-raised here.
            # The worker-side wrapper preserves the worker traceback string on the exception.
            if isinstance(exc, FunctionExecutionError):
                # Log the worker traceback (if present) and re-raise to preserve context
                worker_tb = getattr(exc, "worker_tb", None)
                if worker_tb:
                    # log at error level with the worker traceback
                    logger.error("Exception in worker process:\n%s", worker_tb)
                    # also print to stdout/stderr so it's visible in interactive runs
                    print("Exception in worker process (traceback):")
                    print(worker_tb)
                # re-raise the exception as-is to preserve semantics
                raise
            # Fallback: wrap any other unexpected exception
            raise FunctionExecutionError(
                f"Function raised an exception in worker process: {exc}",
                worker_tb=None
            ) from exc

def _worker_wrapper(func: Callable, args: Tuple, kwargs: Dict[str, Any]):
    """
    Worker wrapper executed inside the child process. Capture any exception,
    attach the formatted traceback string to FunctionExecutionError so it can be
    transmitted to the parent process via pickle, and raise that instead.
    """
    try:
        return func(*args, **(kwargs or {}))
    except Exception as exc:
        # Format full traceback in the worker
        tb = traceback.format_exc()
        msg = f"{exc!r}\n\nWorker traceback:\n{tb}"
        # Raise a picklable exception that carries the worker traceback string
        raise FunctionExecutionError(msg, worker_tb=tb)
