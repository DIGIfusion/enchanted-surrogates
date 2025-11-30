import pytest
from unittest.mock import MagicMock

@pytest.fixture
def fake_sampler():
    """
    Reusable sampler mock for executor unit tests.
    Samples are obtained from the list given to the factory function.
    """
    def _factory(sample_batches: list[list[dict]]):
        sampler = MagicMock()

        # On every function call, next list of dicts from sample_batches ir returned
        sampler.get_next_samples.side_effect = sample_batches

        # Budget ends after all batches have been consumed
        def has_budget():
            return sampler.get_next_samples.call_count < len(sample_batches)

        type(sampler).has_budget = property(lambda self: has_budget())

        return sampler

    return _factory