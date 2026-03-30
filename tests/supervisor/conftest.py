import pytest
import os
import pandas as pd
from unittest.mock import MagicMock


@pytest.fixture
def fake_sampler():
    """
    Reusable sampler mock for supervisor unit tests.
    Samples are obtained from the list given to the factory function.
    """

    def _factory(sample_batches: list[list[dict]]):
        sampler = MagicMock()
        sampler.__name__ = "MockSampler"
        sampler._submitted = 0
        sampler._budget = sum(len(batch) for batch in sample_batches)

        # On every function call, next list of dicts from sample_batches is returned
        def _get_next_samples():
            batch_index = sampler.get_next_samples.call_count - 1
            batch = sample_batches[batch_index]
            sampler._submitted += len(batch)
            return batch

        sampler.get_next_samples.side_effect = _get_next_samples

        def _get_budget(self):
            return self._budget

        def _set_budget(self, value):
            self._budget = value

        type(sampler).budget = property(_get_budget, _set_budget)

        type(sampler).submitted = property(lambda self: self._submitted)

        type(sampler).has_budget = property(lambda self: self._submitted < self._budget)

        return sampler

    return _factory


@pytest.fixture
def fake_executor():
    """
    Reusable executor mock for Supervisor unit tests.
    """

    def _factory():
        executor = MagicMock()
        executor.__name__ = "MockExecutor"

        def execute(input):
            for run_dir, sample in input:
                create_fake_output_csv(run_dir, sample)

        executor.execute.side_effect = execute
        return executor

    return _factory


@pytest.fixture
def patch_supervisor_imports(monkeypatch, fake_sampler, fake_executor):
    def _factory(sampler_batches=None):
        if sampler_batches is None:
            sampler_batches = []

        batch_iter = iter(sampler_batches)
        samplers = []
        executors = []

        def import_sampler(*args, **kwargs):
            sampler = fake_sampler(next(batch_iter, [[{}]]))
            samplers.append(sampler)
            return sampler

        def import_executor(*args, **kwargs):
            executor = fake_executor()
            executors.append(executor)
            return executor

        monkeypatch.setattr(
            "enchanted_surrogates.supervisor.nested_imports.import_sampler",
            import_sampler,
        )
        monkeypatch.setattr(
            "enchanted_surrogates.supervisor.nested_imports.import_executor",
            import_executor,
        )

        return samplers, executors

    return _factory


def create_fake_output_csv(run_dir: str, sample: dict):
    """Mock version of what run_simulation_task does"""
    os.makedirs(run_dir, exist_ok=True)
    output = {"success": True, **sample, "run_dir": run_dir}
    df_point = pd.DataFrame({r: [v] for r, v in output.items()})
    df_point.to_csv(
        os.path.join(run_dir, "enchanted_datapoint.csv"), header=True, index=False
    )
