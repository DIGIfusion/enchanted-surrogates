import os
import numpy as np
import pandas as pd
import pytest

import torch

from enchanted_surrogates.samplers.gpy_torch_active_sampler import GpyTorchActiveSampler


# ============================================================
# 1. Test: NPY iterator reads all rows
# ============================================================
def test_npy_iterator_reads_all_rows(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0", "x1", "x2"],
        bounds=[[0,1]]*3,
        pool_csv_path=None,  # triggers random NPY pool generation
        total_pool_size=50000,
        pool_chunk_size=10000,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    total_seen = 0
    while True:
        X_unit, y, idx = sampler.get_next_pool_chunk()
        if X_unit is None:
            break
        total_seen += len(idx)

    assert total_seen == 50000
    
    sampler.remove_pool()



# ============================================================
# 2. Test: Global indices increment correctly
# ============================================================
def test_global_indices_increment(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0", "x1", "x2"],
        bounds=[[0,1]]*3,
        pool_csv_path=None,
        total_pool_size=30000,
        pool_chunk_size=10000,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    expected_start = 0
    while True:
        X_unit, y, idx = sampler.get_next_pool_chunk()
        if X_unit is None:
            break
        assert idx[0] == expected_start
        expected_start += len(idx)
    
    sampler.remove_pool()



# ============================================================
# 3. Test: Chunk index ranges are contiguous
# ============================================================
def test_chunk_ranges_contiguous(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0", "x1", "x2"],
        bounds=[[0,1]]*3,
        pool_csv_path=None,
        total_pool_size=45000,
        pool_chunk_size=10000,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    prev_end = -1
    while True:
        X_unit, y, idx = sampler.get_next_pool_chunk()
        if X_unit is None:
            break
        if prev_end >= 0:
            assert idx[0] == prev_end + 1
        prev_end = idx[-1]
    
    sampler.remove_pool()



# ============================================================
# 4. Test: Removed indices are skipped
# ============================================================
def test_removed_indices_are_skipped(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0", "x1", "x2"],
        bounds=[[0,1]]*3,
        pool_csv_path=None,
        total_pool_size=20000,
        pool_chunk_size=5000,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    sampler._remove_from_pool({10, 11, 12, 9999})

    seen = set()
    while True:
        X_unit, y, idx = sampler.get_next_pool_chunk()
        if X_unit is None:
            break
        seen.update(idx)

    assert 10 not in seen
    assert 11 not in seen
    assert 12 not in seen
    assert 9999 not in seen
    
    sampler.remove_pool()



# ============================================================
# 5. Test: _get_unit_points_by_global_indices on REAL 10M pool
# ============================================================
def test_get_unit_points_by_global_indices(tmp_path):
    pool_rows = 10_000_000

    sampler = GpyTorchActiveSampler(
        parameters=["x0", "x1", "x2"],
        bounds=[[0,1]]*3,
        pool_csv_path=None,
        total_pool_size=pool_rows,
        pool_chunk_size=200_000,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    target_indices = [int(i) for i in np.linspace(5, pool_rows - 1, 5)]

    pts = sampler._get_unit_points_by_global_indices(target_indices)

    assert pts.shape == (len(target_indices), len(sampler.parameters))
    
    sampler.remove_pool()



# ============================================================
# 6. Test: Large pool index mapping (mocked)
# ============================================================
def test_large_pool_index_mapping(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0", "x1", "x2"],
        bounds=[[0,1]]*3,
        pool_csv_path=None,
        total_pool_size=20000,
        pool_chunk_size=5000,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    sampler.total_pool_size = 10_000_001  # simulate huge pool

    X_unit, y, idx = sampler.get_next_pool_chunk()
    assert idx[0] == 0
    assert idx[-1] == 4999

    X_unit, y, idx = sampler.get_next_pool_chunk()
    assert idx[0] == 5000
    assert idx[-1] == 9999
    
    sampler.remove_pool()



# ============================================================
# 7. Test: Iterator resets correctly (NPY version)
# ============================================================
def test_iterator_resets_correctly(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0","x1"],
        bounds=[[0,1],[0,1]],
        pool_csv_path=None,
        total_pool_size=20000,
        pool_chunk_size=5000,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    seen_first = []
    while True:
        X, y, idx = sampler.get_next_pool_chunk()
        if X is None:
            break
        seen_first.extend(idx)

    # Reset NPY iterator
    sampler._next_row_index = 0

    seen_second = []
    while True:
        X, y, idx = sampler.get_next_pool_chunk()
        if X is None:
            break
        seen_second.extend(idx)

    assert seen_first == seen_second
    
    sampler.remove_pool()



# ============================================================
# 8. Test: Unit-space conversion correctness
# ============================================================
def test_unit_space_conversion(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0","x1"],
        bounds=[[0,10],[100,200]],
        pool_csv_path=None,
        total_pool_size=1000,
        pool_chunk_size=500,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    real = np.array([[5,150],[0,100],[10,200]], dtype=float)
    unit = sampler.to_unit_numpy(real)
    back = sampler.from_unit_numpy(unit)

    assert np.allclose(real, back)
    
    sampler.remove_pool()



# ============================================================
# 9. Test: allowed_pool_values respected
# ============================================================
def test_allowed_pool_values(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0"],
        bounds=[[0,10]],
        allowed_pool_values={"x0":[1,2,3]},
        pool_csv_path=None,
        total_pool_size=5000,
        pool_chunk_size=1000,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    X, y, idx = sampler.get_next_pool_chunk()
    real = sampler.from_unit_numpy(X)

    vals = np.round(real[:,0]).astype(int)
    assert np.all(np.isin(vals, [1,2,3]))
    
    sampler.remove_pool()




# ============================================================
# 10. Test: seed reproducibility
# ============================================================
def test_seed_reproducibility(tmp_path):
    s1 = GpyTorchActiveSampler(
        parameters=["x0","x1"],
        bounds=[[0,1],[0,1]],
        pool_csv_path=None,
        total_pool_size=20000,
        pool_chunk_size=5000,
        base_run_dir=os.path.join(tmp_path, 'run1', "tmp_base_run_dir"),
        output_col="output",
        seed=123,
        remove_pool_file=True,
    )

    s2 = GpyTorchActiveSampler(
        parameters=["x0","x1"],
        bounds=[[0,1],[0,1]],
        pool_csv_path=None,
        total_pool_size=20000,
        pool_chunk_size=5000,
        base_run_dir=os.path.join(tmp_path, 'run2', "tmp_base_run_dir"),
        output_col="output",
        seed=123,
        remove_pool_file=True,
    )

    X1, _, _ = s1.get_next_pool_chunk()
    X2, _, _ = s2.get_next_pool_chunk()

    assert np.allclose(X1, X2)
    
    s1.remove_pool()
    s2.remove_pool()

    
# ============================================================
# 11. Test: acquisition scores monotonic with variance mode
# ============================================================
def test_acquisition_variance_mode(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0", "x1"],
        bounds=[[0,1],[0,1]],
        pool_csv_path=None,
        total_pool_size=2000,
        pool_chunk_size=500,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    # Add simple training data
    sampler.train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=sampler.dtype, device=sampler.device)
    sampler.train_y = torch.tensor([[0.0], [1.0]], dtype=sampler.dtype, device=sampler.device)

    sampler.fit_gpr_model(num_restarts=1)

    sampler.acquisition_mode = "variance"

    X_unit, _, idx = sampler.get_next_pool_chunk()
    scores = sampler.compute_acquisition_scores(X_unit)

    assert scores.shape[0] == X_unit.shape[0]
    assert np.all(scores >= 0.0)
    
    sampler.remove_pool()


# ============================================================
# 12. 
# ============================================================
def test_acquisition_maxPred(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0"],
        bounds=[[0,1]],
        pool_csv_path=None,
        total_pool_size=1000,
        pool_chunk_size=500,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    sampler.train_x = torch.tensor([[0.0], [1.0]], dtype=sampler.dtype, device=sampler.device)
    sampler.train_y = torch.tensor([[0.0], [1.0]], dtype=sampler.dtype, device=sampler.device)

    sampler.fit_gpr_model(num_restarts=1)

    sampler.acquisition_mode = "maxPred"

    X_unit, _, _ = sampler.get_next_pool_chunk()
    scores = sampler.compute_acquisition_scores(X_unit)

    assert scores.shape[0] == X_unit.shape[0]

    sampler.remove_pool()


# ============================================================
# 13. 
# ============================================================

def test_acquisition_eim(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0"],
        bounds=[[0,1]],
        pool_csv_path=None,
        total_pool_size=1000,
        pool_chunk_size=500,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    sampler.train_x = torch.tensor([[0.0], [1.0]], dtype=sampler.dtype, device=sampler.device)
    sampler.train_y = torch.tensor([[0.0], [1.0]], dtype=sampler.dtype, device=sampler.device)

    sampler.fit_gpr_model(num_restarts=1)

    sampler.acquisition_mode = "eim"

    X_unit, _, _ = sampler.get_next_pool_chunk()
    scores = sampler.compute_acquisition_scores(X_unit)

    assert np.all(scores >= 0.0)

    sampler.remove_pool()


# ============================================================
# 14. Test: compute_acquisition_candidates returns correct shape
# ============================================================
def test_compute_acquisition_candidates_shape(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0","x1"],
        bounds=[[0,1],[0,1]],
        pool_csv_path=None,
        total_pool_size=5000,
        pool_chunk_size=1000,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        batch_size=5,
        remove_pool_file=True,
    )

    sampler.train_x = torch.tensor([[0.0,0.0],[1.0,1.0]], dtype=sampler.dtype, device=sampler.device)
    sampler.train_y = torch.tensor([[0.0],[1.0]], dtype=sampler.dtype, device=sampler.device)

    sampler.fit_gpr_model(num_restarts=1)

    X_real, X_unit, idx = sampler.compute_acquisition_candidates()

    assert X_real.shape == (5, 2)
    assert X_unit.shape == (5, 2)
    assert len(idx) == 5
    
    sampler.remove_pool()



# ============================================================
# 15. Test: fantasy batch mode selects distinct points
# ============================================================
def test_compute_acquisition_candidates_fantasy(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0"],
        bounds=[[0,1]],
        pool_csv_path=None,
        total_pool_size=5000,
        pool_chunk_size=1000,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        batch_size=3,
        acquisition_batch_mode="fantasy_labels",
        remove_pool_file=True,
    )

    sampler.train_x = torch.tensor([[0.0],[1.0]], dtype=sampler.dtype, device=sampler.device)
    sampler.train_y = torch.tensor([[0.0],[1.0]], dtype=sampler.dtype, device=sampler.device)

    sampler.fit_gpr_model(num_restarts=1)

    # IMPORTANT: use a real acquisition mode
    sampler.acquisition_mode = "var"

    X_real, X_unit, idx = sampler.compute_acquisition_candidates()

    assert len(idx) == 3
    assert len(set(idx)) == 3

    sampler.remove_pool()
    
# ============================================================
# 16. Test: EIGF acquisition mode
# ============================================================
def test_acquisition_eigf(tmp_path):
    sampler = GpyTorchActiveSampler(
        parameters=["x0", "x1"],
        bounds=[[0,1],[0,1]],
        pool_csv_path=None,
        total_pool_size=3000,
        pool_chunk_size=500,
        base_run_dir=os.path.join(tmp_path, "tmp_base_run_dir"),
        output_col="output",
        remove_pool_file=True,
    )

    # Simple training data
    sampler.train_x = torch.tensor(
        [[0.0, 0.0],
         [1.0, 1.0],
         [0.5, 0.5]],
        dtype=sampler.dtype,
        device=sampler.device
    )
    sampler.train_y = torch.tensor(
        [[0.0],
         [1.0],
         [0.5]],
        dtype=sampler.dtype,
        device=sampler.device
    )

    sampler.fit_gpr_model(num_restarts=1)

    sampler.acquisition_mode = "eigf"

    # Stream one chunk
    X_unit, _, _ = sampler.get_next_pool_chunk()
    scores = sampler.compute_acquisition_scores(X_unit)

    # Basic checks
    assert scores.shape[0] == X_unit.shape[0]
    assert np.all(np.isfinite(scores))
    assert np.any(scores > 0.0)      # EIGF should produce positive values
    assert np.std(scores) > 0.0      # Should not be constant


