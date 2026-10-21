import numpy as np
import pytest

from enchanted_surrogates.samplers.helena_volume_average_sampler import HelenaVolumeAverageSampler

TE_KEYS = {"T_eped", "T_esep", "aT1", "d_T_ped", "alpha_T1", "alpha_T2"}
NE_KEYS = {"n_eped", "n_esep", "an1", "d_n_ped", "alpha_n1", "alpha_n2"}


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
