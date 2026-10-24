import numpy as np
import pytest

from enchanted_surrogates.samplers.helena_volume_average_sampler import HelenaVolumeAverageSampler

TE_KEYS = {"T_eped", "T_esep", "aT1", "d_T_ped", "alpha_T1", "alpha_T2", "Te_avg_target"}
NE_KEYS = {"n_eped", "n_esep", "an1", "d_n_ped", "alpha_n1", "alpha_n2", "ne_avg_target"}


def _profile(phys, letter, psi):
    """idete=7 profile from the &phys entries; letter 't' -> *te, 'd' -> *de."""
    suf = "e" if letter == "t" else "e"
    pre = "te" if letter == "t" else "de"
    a0, sep, a1 = phys["a" + pre], phys["b" + pre], phys["c" + pre]
    mid = phys["ddte"] if letter == "t" else phys["dde"]
    delta, ped = phys["e" + pre], phys["f" + pre]
    al1, al2 = phys["g" + pre], phys["h" + pre]
    t1 = np.tanh(2 * (1 - mid) / delta) - np.tanh(2 * (psi - mid) / delta)
    x = np.clip(psi / ped, 0, 1)
    return sep + a0 * t1 + a1 * np.where(psi <= ped, (1 - x ** al1) ** al2, 0.0)


def _sampler(**kw):
    base = dict(budget=100, te_avg_range=[3.1, 6.3], ne_avg_range=[4.2, 8.1], batch_size=100, seed=0)
    base.update(kw)
    return HelenaVolumeAverageSampler(**base)


def test_keys_and_ranges():
    samples = _sampler().get_next_samples()
    assert len(samples) == 100
    for s in samples:
        assert set(s) == TE_KEYS | NE_KEYS
        assert 0.01 <= s["d_T_ped"] <= 0.3 and 0.01 <= s["d_n_ped"] <= 0.3
        assert 1 <= s["alpha_T1"] <= 2.5
        assert s["T_esep"] < s["T_eped"] and s["n_esep"] < s["n_eped"]
        assert s["aT1"] >= 0 and s["an1"] >= 0


def test_relative_parameters_are_sampled_in_their_ranges():
    smp = _sampler(te_ped_range=[0.2, 0.3], te_sep_range=[0.1, 0.2],
                   ne_ped_range=[0.5, 0.6], ne_sep_range=[0.3, 0.4])
    for s in smp.get_next_samples():
        # core value = sep + a0*(tanh(1)+tanh(2*mid/delta)) + a1; recover T_core
        for pre, ped_r, sep_r, p in (("T_", (0.2, 0.3), (0.1, 0.2), "T"), ("n_", (0.5, 0.6), (0.3, 0.4), "n")):
            delta = s[f"d_{p}_ped"]
            a0 = (s[f"{pre}eped"] - s[f"{pre}esep"]) / (2 * np.tanh(1))
            core = s[f"{pre}esep"] + a0 * (np.tanh(1) + np.tanh(2 * (1 - delta / 2) / delta)) \
                + s["aT1" if p == "T" else "an1"]
            assert ped_r[0] - 1e-9 <= s[f"{pre}eped"] / core <= ped_r[1] + 1e-9
            assert sep_r[0] - 1e-9 <= s[f"{pre}esep"] / s[f"{pre}eped"] <= sep_r[1] + 1e-9


def test_ti_ratios_and_seed():
    kw = dict(ti_te_ped_range=[0.8, 1.2], ti_te_sep_range=[1.0, 3.0])
    a = _sampler(**kw).get_next_samples()
    assert a == _sampler(**kw).get_next_samples()
    assert all(0.8 <= s["Ti_ov_Te_ped"] <= 1.2 and 1.0 <= s["Ti_ov_Te_sep"] <= 3.0 for s in a)
    assert "Ti_ov_Te_ped" not in _sampler().get_next_samples()[0]


def test_single_species():
    s = _sampler(ne_avg_range=None).get_next_samples()[0]
    assert set(s) == TE_KEYS


@pytest.mark.parametrize("kw", [
    dict(te_avg_range=None, ne_avg_range=None),
    dict(te_ped_range=[0.1, 1.5]),                        # relative must be within [0, 1]
    dict(ne_sep_range=[0.5, 0.2]),                        # lo > hi
    dict(bogus_range=[0, 1]),
    dict(ti_te_ped_range=[0.8, 1.2]),                     # needs both ti keys
    dict(te_avg_range=None, ti_te_ped_range=[1, 1], ti_te_sep_range=[1, 1]),   # needs Te
])
def test_validation(kw):
    with pytest.raises(ValueError):
        _sampler(**kw)


def test_impossible_ranges_raise():
    smp = _sampler(te_ped_range=[0.99, 1.0], te_sep_range=[0.0, 0.0], d_T_ped_range=[0.01, 0.02])
    smp.MAX_REDRAWS = 20
    with pytest.raises(RuntimeError):
        smp.get_next_samples()


def test_parameters_attribute_matches_sample_keys():
    smp = _sampler(ti_te_ped_range=[0.8, 1.2], ti_te_sep_range=[1.0, 2.0])
    assert set(smp.parameters) == set(smp.get_next_samples()[0])
    assert set(_sampler(ne_avg_range=None).parameters) == TE_KEYS


def test_works_inside_sample_wise_join_sampler():
    from enchanted_surrogates.samplers.sample_wise_join_sampler import SampleWiseJoinSampler
    join = SampleWiseJoinSampler(samplers={
        "shape": dict(type="random_sampler", parameters=["ellip", "tria"],
                      bounds=[[1.7, 1.9], [0.3, 0.5]], budget=4, batch_size=2),
        "kin": dict(type="helena_volume_average_sampler", budget=4, batch_size=2,
                    te_avg_range=[3.1, 6.3], ne_avg_range=[4.2, 8.1], seed=0),
    })
    out = join.get_next_samples()
    assert len(out) == 2 and {"ellip", "tria", "T_eped", "n_eped"} <= set(out[0])
    assert set(join.parameters) >= {"ellip", "tria", "T_eped", "n_eped"}


def test_sobol_method_is_reproducible_stratified_and_valid():
    kw = dict(budget=256, batch_size=256, seed=4, method="sobol")
    a = _sampler(**kw).get_next_samples()
    assert a == _sampler(**kw).get_next_samples()
    assert a != _sampler(budget=256, batch_size=256, seed=4).get_next_samples()
    assert len(a) == 256 and all(set(s) == TE_KEYS | NE_KEYS for s in a)
    # width is a direct Sobol coordinate: 256 points -> exactly one per 1/256 stratum
    for key, (lo, hi) in (("d_T_ped", (0.01, 0.3)), ("d_n_ped", (0.01, 0.3))):
        counts, _ = np.histogram([s[key] for s in a], bins=256, range=(lo, hi))
        assert counts.max() == 1 and counts.min() == 1


def test_sobol_with_ti_ratios_and_bad_method():
    a = _sampler(method="sobol", batch_size=8, ti_te_ped_range=[0.8, 1.2],
                 ti_te_sep_range=[1.0, 2.0]).get_next_samples()
    assert all(0.8 <= s["Ti_ov_Te_ped"] <= 1.2 for s in a)
    with pytest.raises(ValueError):
        _sampler(method="halton")


# ---- parameters / bounds interface -------------------------------------------------------------------------------------

ALL_NAMES = ["Te_avg", "ne_avg", "Te_ped_ov_core", "Te_sep_ov_ped", "ne_ped_ov_core", "ne_sep_ov_ped",
             "Ti_ov_Te_ped", "Ti_ov_Te_sep", "Ti_ov_Te_core", "d_T_ped", "d_n_ped",
             "alpha_T1", "alpha_T2", "alpha_n1", "alpha_n2"]
ALL_BOUNDS = [[3.1, 6.3], [4.2, 8.1], [0.3, 0.8], [0.02, 0.3], [0.3, 0.8], [0.2, 0.5], [0.8, 1.2], [1.0, 2.0], [0.8, 1.2],
              [0.03, 0.3], [0.03, 0.3], [1.5, 2.5], [1.5, 2.5], [1.5, 2.5], [1.5, 2.5]]


def _new(**kw):
    base = dict(budget=64, batch_size=64, seed=2, method="sobol", parameters=ALL_NAMES, bounds=ALL_BOUNDS)
    base.update(kw)
    return HelenaVolumeAverageSampler(**base)


def test_parameters_and_bounds_match_the_deprecated_range_keywords():
    new = _new(method="random")
    with pytest.warns(DeprecationWarning):
        old = HelenaVolumeAverageSampler(
            budget=64, batch_size=64, seed=2, method="random", te_avg_range=[3.1, 6.3], ne_avg_range=[4.2, 8.1],
            te_ped_range=[0.3, 0.8], te_sep_range=[0.02, 0.3], ne_ped_range=[0.3, 0.8], ne_sep_range=[0.2, 0.5],
            ti_te_ped_range=[0.8, 1.2], ti_te_sep_range=[1.0, 2.0], ti_te_core_range=[0.8, 1.2],
            d_T_ped_range=[0.03, 0.3], d_n_ped_range=[0.03, 0.3], alpha_T_range=[1.5, 2.5], alpha_n_range=[1.5, 2.5])
    assert new.get_next_samples() == old.get_next_samples()
    assert new.parameters == old.parameters


def test_unlisted_parameters_keep_their_defaults():
    smp = HelenaVolumeAverageSampler(budget=20, batch_size=20, seed=0, parameters=["Te_avg"], bounds=[[3, 6]])
    assert set(smp.parameters) == TE_KEYS
    for s in smp.get_next_samples():
        assert 0.01 <= s["d_T_ped"] <= 0.3 and 1 <= s["alpha_T1"] <= 2.5


def test_a_parameter_with_equal_bounds_is_fixed():
    smp = HelenaVolumeAverageSampler(budget=8, batch_size=8, seed=1, method="sobol", parameters=["Te_avg", "d_T_ped"],
                                     bounds=[[4, 4], [0.05, 0.05]])
    assert {s["d_T_ped"] for s in smp.get_next_samples()} == {0.05}


@pytest.mark.parametrize("kw, message", [
    (dict(parameters=["bogus"], bounds=[[1, 2]]), "unknown parameters"),
    (dict(parameters=["Te_avg", "ne_avg"], bounds=[[1, 2]]), "same length"),
    (dict(parameters=["Te_avg", "Te_avg"], bounds=[[1, 2], [1, 2]]), "duplicate"),
    (dict(parameters=["d_T_ped"], bounds=[[0.1, 0.2]]), "Te_avg and/or ne_avg"),
    (dict(parameters=["Te_avg", "Ti_ov_Te_ped"], bounds=[[3, 6], [0.8, 1.2]]), "both Ti_ov_Te_ped and Ti_ov_Te_sep"),
    (dict(parameters=["Te_avg", "Ti_ov_Te_core"], bounds=[[3, 6], [0.8, 1.2]]), "needs Ti_ov_Te_ped"),
    (dict(parameters=["Te_avg", "Te_ped_ov_core"], bounds=[[3, 6], [0.5, 1.5]]), "within"),
])
def test_invalid_parameters_and_bounds_are_rejected(kw, message):
    with pytest.raises(ValueError, match=message):
        HelenaVolumeAverageSampler(budget=4, **kw)


def test_new_and_deprecated_options_cannot_be_mixed():
    with pytest.raises(ValueError, match="do not mix"):
        HelenaVolumeAverageSampler(budget=4, parameters=["Te_avg"], bounds=[[3, 6]], ne_avg_range=[4, 8])


# ---- Ti on the axis ---------------------------------------------------------------------------------------------------

def test_ti_core_ratio_is_within_bounds_and_never_gives_a_hollow_ti_core():
    smp = _new(budget=128, batch_size=128)
    samples = smp.get_next_samples()
    k = 2 * np.tanh(1.0)
    for s in samples:
        assert 0.8 <= s["Ti_ov_Te_core"] <= 1.2
        # Ti core term = Ti(0) - Ti_sep - ati*c must be >= 0 (Ti(0) = ratio * Te(0); Te(0) is the scale of the Te profile)
        c = np.tanh(1.0) + np.tanh(2 * (1 - s["d_T_ped"] / 2) / s["d_T_ped"])
        te_axis = s["T_esep"] + (s["T_eped"] - s["T_esep"]) / k * c + s["aT1"]
        ti_sep, ti_ped = s["Ti_ov_Te_sep"] * s["T_esep"], s["Ti_ov_Te_ped"] * s["T_eped"]
        assert s["Ti_ov_Te_core"] * te_axis - ti_sep - (ti_ped - ti_sep) / k * c >= -1e-9


def test_ti_core_ratio_needs_both_other_ratios():
    smp = _new(parameters=ALL_NAMES, bounds=ALL_BOUNDS)
    assert "Ti_ov_Te_core" in smp.parameters
    s = _new(parameters=[n for n in ALL_NAMES if n != "Ti_ov_Te_core"],
             bounds=[b for n, b in zip(ALL_NAMES, ALL_BOUNDS) if n != "Ti_ov_Te_core"]).get_next_samples()[0]
    assert "Ti_ov_Te_core" not in s and "Ti_ov_Te_ped" in s


# ---- assumed V(psiN)/V_total ----------------------------------------------------------------------------------------

def _batch_of_curves(curve, n, success=True):
    import pandas as pd
    from enchanted_surrogates.utils.volume_scaling import CURVE_COLUMNS
    df = pd.DataFrame([dict(zip(CURVE_COLUMNS, curve)) for _ in range(n)])
    df["success"] = success
    return df


def test_samples_carry_their_volume_average_targets():
    for s in _sampler(batch_size=20).get_next_samples():
        assert 3.1 <= s["Te_avg_target"] <= 6.3 and 4.2 <= s["ne_avg_target"] <= 8.1


def test_the_volume_average_of_a_sample_equals_its_target_with_the_assumed_curve():
    from enchanted_surrogates.utils.volume_scaling import SPECIES, species_average
    smp = _sampler(batch_size=20)
    for s in smp.get_next_samples():
        assert species_average(s, "Te", smp.curve) == pytest.approx(s["Te_avg_target"], rel=1e-3)
        assert species_average(s, "ne", smp.curve) == pytest.approx(s["ne_avg_target"], rel=1e-3)
    assert set(SPECIES["Te"]) <= set(s)


def test_the_curve_is_updated_with_the_solved_equilibria_weighted_with_the_prior():
    from enchanted_surrogates.utils.volume_scaling import MEAN_V_CURVE, KNOTS
    smp = _sampler(volume_prior_weight=10.0)
    assert smp.curve == pytest.approx(MEAN_V_CURVE)
    new = KNOTS ** 2                                             # a different curve
    smp.register_future(_batch_of_curves(new, 10))
    assert smp.curve == pytest.approx(0.5 * MEAN_V_CURVE + 0.5 * new)   # 10 prior + 10 new equilibria
    smp.register_future(_batch_of_curves(new, 20))
    assert smp.curve == pytest.approx((10 * MEAN_V_CURVE + 30 * new) / 40)


def test_failed_or_curveless_rows_do_not_update_the_curve():
    from enchanted_surrogates.utils.volume_scaling import MEAN_V_CURVE, KNOTS
    import pandas as pd
    smp = _sampler()
    smp.register_future(_batch_of_curves(KNOTS ** 2, 5, success=False))
    smp.register_future(pd.DataFrame({"success": [True, True], "a": [1, 2]}))
    smp.register_future(pd.DataFrame())
    assert smp.curve == pytest.approx(MEAN_V_CURVE)


def test_the_updated_curve_scales_the_next_samples():
    from enchanted_surrogates.utils.volume_scaling import KNOTS, species_average
    smp = _sampler(batch_size=5, volume_prior_weight=1.0)
    smp.register_future(_batch_of_curves(KNOTS ** 2, 50))
    for s in smp.get_next_samples():
        assert species_average(s, "Te", smp.curve) == pytest.approx(s["Te_avg_target"], rel=1e-3)


def test_prior_weight_zero_and_a_given_table_keep_the_curve_fixed(tmp_path):
    from enchanted_surrogates.utils.volume_scaling import KNOTS
    smp = _sampler(volume_prior_weight=0)
    before = smp.curve.copy()
    smp.register_future(_batch_of_curves(KNOTS ** 2, 10))
    assert smp.curve == pytest.approx(before)

    table = tmp_path / "v.csv"
    np.savetxt(table, np.c_[KNOTS, KNOTS], delimiter=",")
    smp = _sampler(volume_profile=str(table))
    smp.register_future(_batch_of_curves(KNOTS ** 2, 10))
    assert smp.curve == pytest.approx(KNOTS)


def test_a_join_sampler_passes_the_results_on_so_that_the_curve_is_updated():
    from enchanted_surrogates.samplers.sample_wise_join_sampler import SampleWiseJoinSampler
    from enchanted_surrogates.utils.volume_scaling import KNOTS, MEAN_V_CURVE
    join = SampleWiseJoinSampler(samplers={
        "shape": dict(type="random_sampler", parameters=["ellip", "tria"],
                      bounds=[[1.7, 1.9], [0.3, 0.5]], budget=4, batch_size=2),
        "kin": dict(type="helena_volume_average_sampler", budget=4, batch_size=2,
                    te_avg_range=[3.1, 6.3], ne_avg_range=[4.2, 8.1], seed=0, volume_prior_weight=2.0),
    })
    kinetic = join.all_samplers[1]
    join.register_future(_batch_of_curves(KNOTS ** 2, 2))
    assert kinetic.curve == pytest.approx(0.5 * MEAN_V_CURVE + 0.5 * KNOTS ** 2)
