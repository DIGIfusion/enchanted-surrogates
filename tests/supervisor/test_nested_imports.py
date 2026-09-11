from types import SimpleNamespace
import pytest

from enchanted_surrogates.supervisor.nested_imports import (
    parse_sequential_group,
    import_packers,
    parse_all_run_groups,
    RunGroup,
)


def test_parse_sequential_group_without_packer():
    group_config = {"sampler": "s1", "executor": "e1", "runner": "r1"}
    sampler, executors, runners, packers = parse_sequential_group(group_config)

    assert sampler == "s1"
    assert executors == ["e1"]
    assert runners == ["r1"]
    assert packers is None


def test_parse_sequential_group_with_single_packer():
    group_config = {"sampler": "s1", "executor": "e1", "runner": "r1", "packer": "p1"}
    _, _, _, packers = parse_sequential_group(group_config)

    assert packers == ["p1"]


def test_parse_sequential_group_with_packer_list():
    group_config = {
        "sampler": "s1",
        "executor": ["e1", "e2"],
        "runner": ["r1", "r2"],
        "packer": ["p1", "p2"],
    }
    _, _, _, packers = parse_sequential_group(group_config)

    assert packers == ["p1", "p2"]


def test_import_packers_missing_section_returns_empty_dict():
    # No 'packers' attribute at all - packers are optional at the top level.
    args = SimpleNamespace(executors={}, samplers={}, runners={})

    assert import_packers(args) == {}


def test_import_packers_builds_instances(monkeypatch):
    seen_configs = {}

    def fake_import_packer(packer_type, packer_config):
        seen_configs[packer_type] = packer_config
        return f"packer-instance:{packer_type}"

    monkeypatch.setattr(
        "enchanted_surrogates.supervisor.nested_imports.import_packer",
        fake_import_packer,
    )

    args = SimpleNamespace(
        packers={"p1": {"type": "mock_packer", "hdf5_path": "/tmp/x.h5"}}
    )

    packers = import_packers(args)

    assert packers == {"p1": "packer-instance:mock_packer"}
    assert seen_configs == {"mock_packer": {"type": "mock_packer", "hdf5_path": "/tmp/x.h5"}}


def test_parse_all_run_groups_wires_packers(monkeypatch):
    monkeypatch.setattr(
        "enchanted_surrogates.supervisor.nested_imports.import_executor",
        lambda t, c: f"executor:{t}",
    )
    monkeypatch.setattr(
        "enchanted_surrogates.supervisor.nested_imports.import_sampler",
        lambda t, c: f"sampler:{t}",
    )
    monkeypatch.setattr(
        "enchanted_surrogates.supervisor.nested_imports.import_packer",
        lambda t, c: f"packer:{t}",
    )

    args = SimpleNamespace(
        executors={"e1": {"type": "mock_executor"}},
        samplers={"s1": {"type": "mock_sampler"}},
        runners={"r1": {"type": "mock_runner"}},
        packers={"p1": {"type": "mock_packer"}},
        supervisor={
            "run_order": [
                {"sampler": "s1", "executor": "e1", "runner": "r1", "packer": "p1"},
                {"sampler": "s1", "executor": "e1", "runner": "r1"},
            ]
        },
    )

    groups = parse_all_run_groups(args)

    assert groups[0].packers == ["packer:mock_packer"]
    assert groups[1].packers is None


def test_run_group_validate_raises_on_packer_runner_count_mismatch():
    group = RunGroup(
        executors=["e1"],
        sampler="s1",
        runners=[{"type": "r1"}],
        packers=["p1", "p2"],
    )

    with pytest.raises(ValueError, match="packers and runners"):
        group.validate()


def test_run_group_validate_passes_with_no_packers():
    group = RunGroup(
        executors=["e1"],
        sampler="s1",
        runners=[{"type": "r1"}],
    )

    group.validate()
