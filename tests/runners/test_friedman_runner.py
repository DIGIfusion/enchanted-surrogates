import pytest
import os
import numpy as np
from unittest.mock import patch
from enchanted_surrogates.runners.friedman_runner import FriedmanRunner  # Replace with actual import path

# these tests were generated using gemeni 3 14th march 2026

@pytest.fixture
def runner_config():
    return {
        "sleep_sec": 0.0,
        "noise_std": 0.0,
        "fail_prob": 0.0
    }

@pytest.fixture
def valid_params():
    return {"x1": 0.5, "x2": 0.5, "x3": 0.5, "x4": 0.5, "x5": 0.5}

def test_mathematical_correctness(tmp_path, runner_config, valid_params):
    """Verify y_true calculation matches the Friedman #1 formula."""
    runner = FriedmanRunner(**runner_config)
    run_dir = str(tmp_path / "run_1")
    
    # x1=0.5, x2=0.5 -> sin(pi * 0.25) = sin(45 deg) = 0.7071
    # x3=0.5 -> 20 * (0.5 - 0.5)^2 = 0
    # x4=0.5, x5=0.5 -> 10*0.5 + 5*0.5 = 7.5
    # Expected: 10 * 0.70710678 + 0 + 7.5 = 14.5710678
    
    result = runner.single_code_run(run_dir, params=valid_params)
    
    expected_y = 10 * np.sin(np.pi * 0.5 * 0.5) + 20 * (0.5 - 0.5)**2 + 10 * 0.5 + 5 * 0.5
    assert result["true_value"] == pytest.approx(expected_y)
    assert result["output"] == pytest.approx(expected_y)
    assert result["success"] is True

def test_noise_injection(tmp_path, valid_params):
    """Verify that noise_std > 0 results in output != true_value."""
    runner = FriedmanRunner(noise_std=1.0, sleep_sec=0)
    run_dir = str(tmp_path / "run_noise")
    
    # We run it multiple times to ensure noise is actually random/changing
    results = [runner.single_code_run(run_dir, params=valid_params) for _ in range(5)]
    
    for r in results:
        # true_value should remain constant, output should vary
        assert r["output"] != pytest.approx(r["true_value"], abs=1e-9)
    
    outputs = [r["output"] for r in results]
    assert len(set(outputs)) == 5  # Highly likely all 5 are unique

def test_parameter_validation(tmp_path, runner_config):
    """Verify runner raises errors for missing or non-numeric parameters."""
    runner = FriedmanRunner(**runner_config)
    run_dir = str(tmp_path / "run_fail")

    # Missing keys
    with pytest.raises(ValueError, match="missing required parameters"):
        runner.single_code_run(run_dir, params={"x1": 0.1})

    # Non-numeric keys
    with pytest.raises(TypeError, match="requires numeric inputs"):
        runner.single_code_run(run_dir, params={f"x{i}": "not_a_number" for i in range(1, 6)})

def test_file_io_behavior(tmp_path, runner_config, valid_params):
    """Verify that the runner actually writes and reads from the disk."""
    runner = FriedmanRunner(**runner_config)
    run_dir = tmp_path / "io_test"
    
    runner.single_code_run(str(run_dir), params=valid_params)
    
    outfile = run_dir / "output.txt"
    assert outfile.exists()
    
    with open(outfile, "r") as f:
        content = float(f.read().strip())
        assert content == pytest.approx(14.5710678)

def test_sleep_behavior_range(runner_config):
    """Verify get_sleep_sec handles both scalars and ranges."""
    # Test scalar
    runner_scalar = FriedmanRunner(sleep_sec=0.5)
    assert runner_scalar.get_sleep_sec() == 0.5
    
    # Test range
    runner_range = FriedmanRunner(sleep_sec=[1.0, 2.0])
    for _ in range(10):
        val = runner_range.get_sleep_sec()
        assert 1.0 <= val <= 2.0

    # Test invalid range
    runner_bad = FriedmanRunner(sleep_sec=[5.0, 1.0])
    with pytest.raises(ValueError):
        runner_bad.get_sleep_sec()

def test_failure_injection(tmp_path, valid_params):
    """Verify that fail_prob=1.0 consistently raises RuntimeError."""
    runner = FriedmanRunner(fail_prob=1.0, sleep_sec=0)
    run_dir = str(tmp_path / "run_crash")
    
    with pytest.raises(RuntimeError, match="Synthetic failure injected"):
        runner.single_code_run(run_dir, params=valid_params)