import importlib
from types import SimpleNamespace
import pytest
from enchanted_surrogates.utils.precise_imports import (
    clear_import_cache, cached_import, import_executor, import_sampler, import_runner
)
from enchanted_surrogates.executors import LocalExecutor
from enchanted_surrogates.samplers.random_sampler import RandomSampler
from enchanted_surrogates.runners.example_runner import ExampleRunner

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
        )
    ],
    ids=["executor", "sampler", "runner"]
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
        "enchanted_surrogates.utils.precise_imports.load_plugins",
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
