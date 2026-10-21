from abc import ABC, abstractmethod
from enchanted_surrogates.utils.logger import get_logger

log = get_logger(__name__)


class Executor(ABC):
    def __init__(self, output_dir=None, *args, **kwargs):
        self.output_dir = output_dir  # TODO rename

    @abstractmethod
    def execute(self, samples, runner_config):
        raise NotImplementedError("execute method not implemented.")

    @abstractmethod
    def clean(self):
        raise NotImplementedError("clean method not implemented.")

    def is_tasks_running(self) -> bool:
        """
        True while the executor still has tasks that are waiting, running or otherwise unfinished, False when there are
        none. The supervisor uses it to notice runs that were lost (the task disappeared, for example because its worker
        was killed) and will never write a result.

        The default is True, which means "cannot tell": the supervisor then keeps waiting as it always did. Executors
        that can tell should override this.
        """
        return True
