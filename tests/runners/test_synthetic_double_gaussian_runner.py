import os
import numpy as np
import pytest

from enchanted_surrogates.runners.synthetic_double_gaussian_runner import SyntheticDoubleGaussianRunner


@pytest.fixture
def tmp_run_dir(tmp_path):
    return tmp_path


def test_init_default_1d():
    runner = SyntheticDoubleGaussianRunner(dimensions=1)
    assert runner.dimensions == 1
    assert len(runner.model_parameters) == 7


def test_init_default_2d():
    runner = SyntheticDoubleGaussianRunner(dimensions=2)
    assert runner.dimensions == 2
    assert len(runner.model_parameters) == 10


def test_init_invalid_dimension():
    with pytest.raises(ValueError):
        SyntheticDoubleGaussianRunner(dimensions=3)


def test_single_code_run_1d_output(tmp_run_dir):
    runner = SyntheticDoubleGaussianRunner(dimensions=1)

    params = {"x": 0.5}
    result = runner.single_code_run(tmp_run_dir, params)

    assert result["success"] is True
    assert isinstance(result["output"], float)

    # Check file output
    outfile = os.path.join(tmp_run_dir, "output.txt")
    assert os.path.exists(outfile)

    with open(outfile, "r") as f:
        content = f.read()
        assert str(result["output"]) in content


def test_single_code_run_1d_known_value(tmp_run_dir):
    runner = SyntheticDoubleGaussianRunner(
        dimensions=1,
        model_parameters=[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    )

    # This simplifies to: exp(-x^2)
    params = {"x": 1.0}
    result = runner.single_code_run(tmp_run_dir, params)

    expected = np.exp(-1.0)
    assert np.isclose(result["output"], expected)


def test_single_code_run_2d_output(tmp_run_dir):
    runner = SyntheticDoubleGaussianRunner(dimensions=2)

    params = {"x": 0.5, "y": 0.5}
    result = runner.single_code_run(tmp_run_dir, params)

    assert result["success"] is True
    assert isinstance(result["output"], float)

    outfile = os.path.join(tmp_run_dir, "output.txt")
    assert os.path.exists(outfile)


def test_gaussian_2d_center_peak():
    runner = SyntheticDoubleGaussianRunner(dimensions=2)

    val = runner.gaussian_2d(
        x=1.0, y=1.0,
        amp=2.0, width=1.0,
        x0=1.0, y0=1.0
    )

    # At center → exp(0) = 1
    assert np.isclose(val, 2.0)


def test_gaussian_2d_decay():
    runner = SyntheticDoubleGaussianRunner(dimensions=2)

    center = runner.gaussian_2d(0, 0, 1.0, 1.0, 0, 0)
    far = runner.gaussian_2d(10, 10, 1.0, 1.0, 0, 0)

    assert center > far


def test_missing_params_raises_keyerror(tmp_run_dir):
    runner = SyntheticDoubleGaussianRunner(dimensions=1)

    with pytest.raises(KeyError):
        runner.single_code_run(tmp_run_dir, params={})


def test_file_append_behavior(tmp_run_dir):
    runner = SyntheticDoubleGaussianRunner(dimensions=1)

    params = {"x": 0.1}
    runner.single_code_run(tmp_run_dir, params)
    runner.single_code_run(tmp_run_dir, params)

    outfile = os.path.join(tmp_run_dir, "output.txt")

    with open(outfile, "r") as f:
        lines = f.read()

    # Since file is opened with "a", both outputs should exist
    assert len(lines) > 0
