from abc import ABC, abstractmethod


class Sampler(ABC):
    _budget: int = 100
    _submitted: int = 0

    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def get_next_samples(self) -> list[dict]:
        """
        Should increment SUBMITTED and return a list of new samples to be run.
        """
        raise NotImplementedError("get_next_samples method not implemented.")

    @abstractmethod
    def register_future(self, future):
        """
        Should register a future from a submitted sample.
        """
        raise NotImplementedError("register_future method not implemented.")

    def register_futures(self, futures):
        """
        Register multiple completed evaluations.

        Parameters
        ----------
        futures : iterable
            Iterable of tuples or dicts accepted by `register_future`.
        """
        for f in futures:
            self.register_future(f)

    def skip(self, index: int):
        """
        Allows setting sampler state.
        Used to allow program restarts.
        """
        # Just run get_next_samples 'index' times
        for _ in range(index):
            if self.has_budget:
                self.get_next_samples()

    @property
    def has_budget(self) -> bool:
        return self.submitted < self.budget

    @property
    def submitted(self) -> int:
        return self._submitted

    @submitted.setter
    def submitted(self, value):
        self._submitted = value

    @property
    def budget(self) -> int:
        return self._budget

    @budget.setter
    def budget(self, value):
        self._budget = value
