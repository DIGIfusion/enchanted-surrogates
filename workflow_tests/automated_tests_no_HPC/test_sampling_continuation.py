"""
Full workflow tests for seamless sampling continuation.
"""
from enchanted_surrogates.supervisor.supervisor import Supervisor
from workflow_tests.utils.test_utils import get_run_dir_count

def test_simple_resume_after_interruption(tmp_path, run_config):
    supervisor: Supervisor = run_config(
        "test_configs/simple_resume_after_interruption.yaml",
        call_start = False
    )

    try:
        supervisor.start()
        assert False  # This shouldn't happen
    except Exception:
        pass  # This should happen

    # Read the parameters used for asserts
    sampler = supervisor.args.samplers["random_kill"]
    kill_after: int = sampler["kill_after"]
    batch_size: int = sampler["batch_size"]
    budget: int = sampler["budget"]

    data_dir = tmp_path / "data"

    assert get_run_dir_count(data_dir) == kill_after * batch_size

    supervisor.args.supervisor["run_mode"] = "resume"  # continue where the previous one left off
    supervisor.args.samplers["random_kill"]["kill_after"] = None  # don't kill the poor sampler

    # Create new supervisor from the old args with some changes
    supervisor2 = Supervisor(supervisor.args)
    supervisor2.start()

    assert get_run_dir_count(data_dir) == budget

def test_nested_resume_after_interruption(tmp_path, run_config):
    supervisor: Supervisor = run_config(
        "test_configs/nested_resume_after_interruption.yaml",
        call_start = False
    )

    try:
        supervisor.start()
        assert False  # This shouldn't happen
    except Exception:
        pass  # This should happen

    # Read the parameters used for asserts
    sampler = supervisor.args.samplers["random_kill"]
    kill_after: int = sampler["kill_after"]
    batch_size: int = sampler["batch_size"]

    budget_1: int = supervisor.args.samplers["random"]["budget"]
    budget_2: int = sampler["budget"]
    
    data_dir = tmp_path / "data"

    assert get_run_dir_count(data_dir) == budget_1 + budget_1 * kill_after * batch_size

    supervisor.args.supervisor["run_mode"] = "resume"  # continue where the previous one left off
    supervisor.args.samplers["random_kill"]["kill_after"] = None  # don't kill the poor sampler

    # Create new supervisor from the old args with some changes
    supervisor2 = Supervisor(supervisor.args)
    supervisor2.start()

    assert get_run_dir_count(data_dir) == budget_1 + budget_1 * budget_2

def test_simple_resume_with_increased_budget(tmp_path, run_config):
    supervisor: Supervisor = run_config("test_configs/simple_budget_increase.yaml")

    old_budget = supervisor.args.samplers["random"]["budget"]
    new_budget = old_budget + 10

    data_dir = tmp_path / "data"

    assert get_run_dir_count(data_dir) == old_budget

    # Increase budget and re-run with resume, (new_budget - old_budget) new samples should be got
    supervisor.args.supervisor["run_mode"] = "resume"
    supervisor.args.samplers["random"]["budget"] = new_budget  # increase budget

    supervisor2 = Supervisor(supervisor.args)
    supervisor2.start()

    assert get_run_dir_count(data_dir) == new_budget

def test_simple_extend(tmp_path, run_config):
    supervisor: Supervisor = run_config("test_configs/simple_budget_increase.yaml")

    old_budget = supervisor.args.samplers["random"]["budget"]
    extend_budget_by = 10

    data_dir = tmp_path / "data"

    assert get_run_dir_count(data_dir) == old_budget

    # Set budget and re-run with extend, (extend_budget_by) new samples should be got
    supervisor.args.supervisor["run_mode"] = "extend"
    supervisor.args.samplers["random"]["budget"] = extend_budget_by  # increase budget

    supervisor2 = Supervisor(supervisor.args)
    supervisor2.start()

    assert get_run_dir_count(data_dir) == old_budget + extend_budget_by

def test_nested_extend(tmp_path, run_config):
    supervisor: Supervisor = run_config("test_configs/nested_budget_increase.yaml")

    first_budget = supervisor.args.samplers["s1"]["budget"]
    second_budget = supervisor.args.samplers["s2"]["budget"]
    extend_budget_by = 3

    data_dir = tmp_path / "data"

    assert get_run_dir_count(data_dir) == first_budget + first_budget * second_budget

    # Set budget and re-run with extend, (extend_budget_by) new samples should be got
    supervisor.args.supervisor["run_mode"] = "extend"
    supervisor.args.samplers["s2"]["budget"] = extend_budget_by

    supervisor2 = Supervisor(supervisor.args)
    supervisor2.start()

    assert get_run_dir_count(data_dir) == (
        first_budget + first_budget * (second_budget + extend_budget_by)
    )
