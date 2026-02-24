import numpy as np
import pytest

from ..utils.append_es_to_path import append_es_to_path

append_es_to_path()

from enchanted_surrogates.samplers.bayesian_optimization_sampler import BayesianOptimizationSampler


class DummyParser:
    def __init__(self, mapping):
        self.mapping = mapping

    def collect_sample_information(self, run_dir, observations):
        name = run_dir.split("/")[-1]
        return self.mapping[name]


def test_build_result_dictionary_smoke(tmp_path):
    sampler = BayesianOptimizationSampler(
        bounds=[[0, 10], [0, 20]],
        parameters=["x", "y"],
        parser=None,
    )

    mapping = {
        "run_1": {
            "inputs": np.array([1.0, 2.0]),
            "distances": np.array([3.0, 4.0]),
            "failure": 0,
        },
        "run_fail": {
            "inputs": np.array([9.0, 19.0]),
            "distances": np.array([99.0, 99.0]),
            "failure": 1,
        },
    }
    sampler.parser = DummyParser(mapping)

    (tmp_path / "run_1").mkdir()
    (tmp_path / "run_fail").mkdir()

    sampler.build_result_dictionary(str(tmp_path))

    assert np.asarray(sampler.result_dictionary["inputs"]).shape == (1, 2)
    assert np.asarray(sampler.result_dictionary_failed["inputs"]).shape == (1, 2)
    assert (tmp_path / "result_dictionary.pkl").exists()


def test_train_surrogate_smoke_real_libs():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")

    sampler = BayesianOptimizationSampler(
        bounds=[[0, 10], [0, 20]],
        parameters=["x", "y"],
        parser=None,
        fully_bayesian=False,
    )

    sampler.result_dictionary = {
        "inputs": np.array([[1.0, 2.0], [4.0, 8.0]]),
        "failure": np.array([0.0, 0.0]),
    }
    sampler.result_dictionary_norm = {
        "inputs": np.array([[0.1, 0.1], [0.4, 0.4]]),
        "distances": np.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    sampler.result_dictionary_failed = {
        "inputs": np.array([[10.0, 20.0]]),
        "failure": np.array([1]),
    }

    sampler.train_surrogate()

    assert sampler.model is not None
    assert sampler.model_failed is not None
    assert np.isfinite(float(sampler.best_f))
