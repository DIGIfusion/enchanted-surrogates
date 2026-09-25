import importlib
from types import SimpleNamespace
import pytest
import enchanted_surrogates.utils.precise_imports as precise_imports
from enchanted_surrogates.utils.precise_imports import (
    clear_import_cache, cached_import, import_executor, import_sampler, import_runner,
    import_packer
)
from enchanted_surrogates.executors import LocalExecutor
from enchanted_surrogates.samplers.random_sampler import RandomSampler
from enchanted_surrogates.runners.example_runner import ExampleRunner
from enchanted_surrogates.packers.ascii_bin_to_hdf5_packer import AsciiBinToHdf5Packer

@pytest.mark.parametrize(
    "import_function, type_name, config, expected_type",
    [
        (
            import_executor,
            "LocalExecutor",
            {},
            LocalExecutor
        ),
        (
            import_sampler,
            "RandomSampler",
            {"bounds": [], "budget": 0, "parameters": []},
            RandomSampler
        ),
        (
            import_runner,
            "ExampleRunner",
            {},
            ExampleRunner
        ),
        (
            import_packer,
            "AsciiBinToHdf5Packer",
            {},
            AsciiBinToHdf5Packer
        )
    ],
    ids=["executor", "sampler", "runner", "packer"]
)
def test_import_instantiates_new_objects(
    import_function,
    type_name,
    config,
    expected_type,
):
    clear_import_cache()

    obj_a = import_function(type_name, config)
    obj_b = import_function(type_name, config)

    assert isinstance(obj_a, expected_type)
    assert isinstance(obj_b, expected_type)
    assert obj_a is not obj_b

    clear_import_cache()

def test_cached_import_caches_results(monkeypatch):
    class Mock:
        pass

    mock_module = SimpleNamespace(Mock=Mock)
    import_calls = []
    def mock_import_module(module_name):
        import_calls.append(module_name)
        return mock_module

    # patch importlib.import_module to always return the Mock class
    monkeypatch.setattr(
        importlib, "import_module", mock_import_module
    )
    monkeypatch.setattr(
        precise_imports,
        "load_plugins",
        lambda: {}
    )

    clear_import_cache()
    cls_a = cached_import("Mock", "mock")
    cls_b = cached_import("Mock", "mock")

    # Imported classes are not instances but just the class types
    assert cls_a is Mock
    assert cls_b is Mock
    assert len(import_calls) == 1

    clear_import_cache()


class _KwargsSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _StrictSampler:
    def __init__(self, budget):
        self.budget = budget


@pytest.mark.parametrize(
    "sampler_cls, config, expected_base_run_dir",
    [
        (_KwargsSampler, {"type": "s"}, "/runs/supervisor"),
        (_KwargsSampler, {"type": "s", "base_run_dir": "/runs/supervisor"}, "/runs/supervisor"),
        (_KwargsSampler, {"type": "s", "base_run_dir": "/runs/other"}, "/runs/other"),
    ],
    ids=["inherited", "same_as_supervisor", "config_overrides"],
)
def test_import_sampler_passes_supervisor_base_run_dir(monkeypatch, sampler_cls, config, expected_base_run_dir):
    monkeypatch.setattr(precise_imports, "cached_import", lambda t, m: sampler_cls)
    sampler = import_sampler("s", config, base_run_dir="/runs/supervisor")
    assert sampler.kwargs["base_run_dir"] == expected_base_run_dir
    assert "type" not in sampler.kwargs


def test_import_sampler_warns_on_conflicting_base_run_dir(monkeypatch):
    monkeypatch.setattr(precise_imports, "cached_import", lambda t, m: _KwargsSampler)
    warnings = []
    monkeypatch.setattr(precise_imports.log, "warning", lambda msg: warnings.append(msg))
    import_sampler("s", {"base_run_dir": "/runs/other"}, base_run_dir="/runs/supervisor")
    assert len(warnings) == 1 and "/runs/other" in warnings[0]


def test_import_sampler_leaves_samplers_without_base_run_dir_untouched(monkeypatch):
    monkeypatch.setattr(precise_imports, "cached_import", lambda t, m: _StrictSampler)
    sampler = import_sampler("s", {"type": "s", "budget": 3}, base_run_dir="/runs/supervisor")
    assert sampler.budget == 3
    assert not hasattr(sampler, "base_run_dir")


def test_import_sampler_without_base_run_dir_is_unchanged(monkeypatch):
    monkeypatch.setattr(precise_imports, "cached_import", lambda t, m: _KwargsSampler)
    sampler = import_sampler("s", {"type": "s", "budget": 3})
    assert sampler.kwargs == {"budget": 3}
