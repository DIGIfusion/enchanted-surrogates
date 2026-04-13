import os
import pytest
import numpy as np
from unittest.mock import patch

from enchanted_surrogates.runners.example_runner import ExampleRunner


# ----------------------------------------------------------------------
# Core sleep‑sec behavior (minimal essential coverage)
# ----------------------------------------------------------------------

def test_sleep_sec_fixed_number():
    r = ExampleRunner(sleep_sec=0.25)
    assert r.get_sleep_sec() == 0.25


def test_sleep_sec_random_bounds_valid():
    r = ExampleRunner(sleep_sec=(0.1, 0.2))
    v = r.get_sleep_sec()
    assert 0.1 <= v <= 0.2


def test_sleep_sec_invalid_type():
    r = ExampleRunner(sleep_sec="bad")
    with pytest.raises(TypeError):
        r.get_sleep_sec()


# ----------------------------------------------------------------------
# Core parameter parsing (minimal essential coverage)
# ----------------------------------------------------------------------

def test_single_code_run_two_params(tmp_path):
    r = ExampleRunner()
    out = r.single_code_run(str(tmp_path), params={"a": 3, "b": 7})
    assert out["output"] == 10.0
    assert out["success"] is True


def test_single_code_run_non_numeric_param(tmp_path):
    r = ExampleRunner()
    with pytest.raises(TypeError):
        r.single_code_run(str(tmp_path), params={"a": "not numeric"})


# ----------------------------------------------------------------------
# Core file I/O behavior (minimal essential coverage)
# ----------------------------------------------------------------------

def test_output_file_appends(tmp_path):
    r = ExampleRunner()
    run_dir = str(tmp_path)

    r.single_code_run(run_dir, params={"a": 1})
    r.single_code_run(run_dir, params={"a": 2})

    with open(os.path.join(run_dir, "output.txt"), "r") as f:
        contents = f.read().strip()

    assert "1.0" in contents
    assert "2.0" in contents


# ----------------------------------------------------------------------
# Synthetic failure injection (first 3 tests kept exactly)
# ----------------------------------------------------------------------

def test_fail_prob_zero_never_fails(tmp_path):
    r = ExampleRunner(fail_prob=0.0)
    out = r.single_code_run(str(tmp_path), params={"a": 1})
    assert out["success"] is True


@patch("numpy.random.uniform", return_value=0.0)
def test_fail_prob_triggers_exception_when_raise_true(mock_rand, tmp_path):
    r = ExampleRunner(fail_prob=1.0, raise_failures=True)
    with pytest.raises(RuntimeError):
        r.single_code_run(str(tmp_path), params={"a": 1})


@patch("numpy.random.uniform", return_value=0.0)
def test_fail_prob_sets_success_false_when_raise_false(mock_rand, tmp_path):
    r = ExampleRunner(fail_prob=1.0, raise_failures=False)
    out = r.single_code_run(str(tmp_path), params={"a": 1})
    assert np.isnan(out["output"])
    assert out["success"] is False
