import numpy as np
import pytest

from enchanted_surrogates.utils.volume_scaling import (
    CURVE_COLUMNS, KNOTS, MEAN_V_CURVE, SPECIES, mtanh_profile, species_average, v_curve, volume_weights)


def test_the_mean_curve_is_a_volume_fraction_curve():
    assert len(MEAN_V_CURVE) == len(KNOTS) == len(CURVE_COLUMNS)
    assert MEAN_V_CURVE[0] == 0 and MEAN_V_CURVE[-1] == 1
    assert np.all(np.diff(MEAN_V_CURVE) > 0)


def test_volume_weights_sum_to_one_and_follow_the_curve():
    psi, dv = volume_weights(MEAN_V_CURVE)
    assert dv.sum() == pytest.approx(1.0)
    assert dv[psi < 0.5].sum() == pytest.approx(np.interp(0.5, KNOTS, MEAN_V_CURVE), abs=1e-3)


def test_a_flat_profile_has_its_value_as_average():
    flat = dict(zip(SPECIES["Te"], (2.0, 2.0, 0.0, 0.1, 1.5, 1.5)))   # ped = sep, no core term
    assert species_average(flat, "Te", MEAN_V_CURVE) == pytest.approx(2.0)


def test_profile_hits_its_pedestal_top_separatrix_and_axis_values():
    ped, sep, core, delta = 3.0, 0.5, 4.0, 0.1
    psi = np.array([0.0, 1.0 - delta, 1.0])
    f = mtanh_profile(psi, ped, sep, core, delta, 1.5, 1.5)
    assert f[2] == pytest.approx(sep, abs=0.05 * (ped - sep))             # separatrix
    assert f[0] > f[1] > f[2]                                              # core > pedestal top > separatrix


def test_the_average_is_linear_in_the_profile_amplitude():
    base = dict(zip(SPECIES["ne"], (2.0, 0.5, 3.0, 0.1, 1.5, 1.5)))
    twice = dict(base, n_eped=4.0, n_esep=1.0, an1=6.0)
    assert species_average(twice, "ne", MEAN_V_CURVE) == pytest.approx(2 * species_average(base, "ne", MEAN_V_CURVE))


def test_v_curve_of_nested_ellipses_is_the_identity(tmp_path):
    """psi = rho^2 on ellipses of revolution: V = 2 pi^2 R0 kappa a^2 rho^2, so V/V_total = psiN."""
    freeqdsk = pytest.importorskip("freeqdsk")
    pytest.importorskip("contourpy")
    from freeqdsk import geqdsk
    nx, ny, r0, a, kappa = 129, 257, 3.0, 1.0, 1.7
    R = np.linspace(r0 - 1.2 * a, r0 + 1.2 * a, nx)
    Z = np.linspace(-1.2 * kappa * a, 1.2 * kappa * a, ny)
    rr, zz = np.meshgrid(R, Z, indexing="ij")
    psi = ((rr - r0) / a) ** 2 + (zz / (kappa * a)) ** 2                    # 0 on axis, 1 on the boundary ellipse
    th = np.linspace(0, 2 * np.pi, 201)
    data = dict(nx=nx, ny=ny, rdim=R[-1] - R[0], zdim=Z[-1] - Z[0], rcentr=r0, rleft=R[0], zmid=0.0, rmagx=r0, zmagx=0.0,
                simagx=0.0, sibdry=1.0, bcentr=2.0, cpasma=1e6, fpol=np.ones(nx), pres=np.ones(nx), qpsi=np.ones(nx),
                psi=psi, ffprime=np.ones(nx), pprime=np.ones(nx), rbdry=r0 + a * np.cos(th), zbdry=kappa * a * np.sin(th),
                rlim=np.array([R[0], R[-1]]), zlim=np.array([Z[0], Z[-1]]))
    path = tmp_path / "eqdsk"
    with open(path, "w") as f:
        geqdsk.write(data, f)
    curve = v_curve(str(path))
    assert curve == pytest.approx(KNOTS, abs=0.01)
