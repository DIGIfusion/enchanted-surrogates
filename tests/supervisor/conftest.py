import pytest
from unittest.mock import MagicMock

@pytest.fixture
def fake_sampler():
    """
    Reusable sampler mock for supervisor unit tests.
    Samples are obtained from the list given to the factory function.
    """
    def _factory(sample_batches: list[list[dict]]):
        sampler = MagicMock()

        # On every function call, next list of dicts from sample_batches is returned
        sampler.get_next_samples.side_effect = sample_batches

        # Budget ends after all batches have been consumed
        type(sampler).has_budget = property(
            lambda self: self.get_next_samples.call_count < len(sample_batches)
        )

        return sampler

    return _factory

@pytest.fixture
def fake_executor():
    """
    Reusable executor mock for Supervisor unit tests.
    """
    def _factory():
        executor = MagicMock()
        return executor

    return _factory

@pytest.fixture
def patch_supervisor_imports(monkeypatch, fake_sampler, fake_executor):
    def _factory(sample_batches=None):
        if sample_batches is None:
            sample_batches = []

        sampler = fake_sampler(sample_batches)
        executor = fake_executor()

        monkeypatch.setattr(
            "enchanted_surrogates.supervisor.supervisor.import_sampler",
            lambda *args, **kwargs: sampler,
        )
        monkeypatch.setattr(
            "enchanted_surrogates.supervisor.supervisor.import_executor",
            lambda *args, **kwargs: executor,
        )

        return sampler, executor

    return _factory
