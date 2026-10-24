"""Volume averages of HELENA mtanh profiles and the volume V(psiN) enclosed by the flux surfaces of a solved equilibrium.

HelenaVolumeAverageSampler scales its Te and ne profiles so that their volume averages hit sampled targets. The volume
average weights the profile with dV(psiN), which is only known after the equilibrium is solved, so the sampler has to
assume V(psiN)/V_total (here MEAN_V_CURVE, updated with the equilibria that are already solved). HelenaRunner uses the solved
equilibrium to correct the scaling. Both use the functions below.

V(psiN) = pi * closed integral of R^2 dZ over the flux surface psiN (Pappus), psiN = (psi - psi_axis)/(psi_bdry - psi_axis).
"""
import numpy as np

# V(psiN)/V_total is stored at these psiN values (a curve is an array of len(KNOTS) values from 0 to 1)
KNOTS = np.linspace(0.0, 1.0, 21)

# Dataset columns that carry the V(psiN)/V_total curve of a solved equilibrium (HelenaRunner output), one per knot
CURVE_COLUMNS = [f"V_over_Vtot_{i:02d}" for i in range(len(KNOTS))]

# Mean V(psiN)/V_total of the 115 converged equilibria of the first JT-60SA production run (geometry of all 5 scenarios,
# broad profile ranges). The fixed table used before (one scenario-#2 run) gave a volume average 15% too low, because
# its V(psiN) is 0.27 at psiN = 0.5 where these equilibria have 0.45 +- 0.06.
MEAN_V_CURVE = np.array([0.0, 0.04364, 0.08728, 0.13108, 0.17507, 0.21923, 0.2636, 0.30827, 0.35335, 0.399, 0.44543,
                         0.49285, 0.54151, 0.59169, 0.64367, 0.69771, 0.75394, 0.81225, 0.87241, 0.93443, 1.0])

# The same constant as helena_parser.TANH_NORMALIZER = tanh(1) - tanh(-1)
TANH_K = 2.0 * np.tanh(1.0)

# Names of the mtanh parameters of each species: ped (value at the pedestal top), sep, core (core term), pedestal width,
# the two core exponents. Te and ne are the species the volume average is scaled for.
SPECIES = {
    "Te": ("T_eped", "T_esep", "aT1", "d_T_ped", "alpha_T1", "alpha_T2"),
    "ne": ("n_eped", "n_esep", "an1", "d_n_ped", "alpha_n1", "alpha_n2"),
}


def mtanh_profile(psi, ped, sep, core, delta, alpha1, alpha2):
    """HELENA idete=7 profile at psiN = psi: sep + (ped - sep)/K * t1 + core * t2, with the tanh edge t1 of width delta
    centred at 1 - delta/2 and the core shape t2 = (1 - (psiN/(1 - delta))^alpha1)^alpha2 inside the pedestal top."""
    psi = np.asarray(psi, dtype=float)
    mid, ped_psi = 1.0 - delta / 2.0, 1.0 - delta
    t1 = np.tanh(2.0 * (1.0 - mid) / delta) - np.tanh(2.0 * (psi - mid) / delta)
    x = np.clip(psi / ped_psi, 0.0, 1.0)
    t2 = np.where(psi <= ped_psi, (1.0 - x ** alpha1) ** alpha2, 0.0)
    return sep + (ped - sep) / TANH_K * t1 + core * t2


def volume_weights(curve, n=2000):
    """(psiN at the cell centres, dV/V_total of each cell) for n cells of equal psiN width, from a V/V_total curve at KNOTS."""
    psi = np.linspace(0.0, 1.0, n + 1)
    dv = np.diff(np.interp(psi, KNOTS, curve))
    return 0.5 * (psi[1:] + psi[:-1]), dv / dv.sum()


def species_average(params, species, curve):
    """Volume average of the Te or ne mtanh profile given by `params` (keys of SPECIES[species])."""
    psi, dv = volume_weights(curve)
    return float(np.sum(mtanh_profile(psi, *(params[k] for k in SPECIES[species])) * dv))


def volume_profile(eqdsk, n_levels=100):
    """dict(psin, v [m^3], v_total) from a G-EQDSK file, psin/v include the anchors (0, 0) and (1, v_total).
    Needs contourpy and freeqdsk."""
    import contourpy
    from freeqdsk import geqdsk
    with open(eqdsk) as f:
        g = geqdsk.read(f)
    R = np.linspace(g["rleft"], g["rleft"] + g["rdim"], g["nx"])
    Z = np.linspace(g["zmid"] - g["zdim"] / 2, g["zmid"] + g["zdim"] / 2, g["ny"])
    pn = (np.asarray(g["psi"]) - g["simagx"]) / (g["sibdry"] - g["simagx"])      # psi[i, j] at (R[i], Z[j])
    gen = contourpy.contour_generator(R, Z, pn.T)
    levels = np.linspace(0.005, 0.995, n_levels)
    v = np.zeros(n_levels)
    for k, lev in enumerate(levels):
        for line in gen.lines(lev):
            if len(line) > 10 and np.hypot(*(line[0] - line[-1])) < 1e-6:        # a closed surface
                v[k] = max(v[k], _pappus(line[:, 0], line[:, 1]))
    rb, zb = np.asarray(g["rbdry"]), np.asarray(g["zbdry"])
    v_total = _pappus(np.r_[rb, rb[0]], np.r_[zb, zb[0]])
    return dict(psin=np.r_[0.0, levels, 1.0], v=np.r_[0.0, v, v_total], v_total=float(v_total))


def v_curve(eqdsk):
    """V(psiN)/V_total of a G-EQDSK at KNOTS."""
    vp = volume_profile(eqdsk)
    return np.interp(KNOTS, vp["psin"], vp["v"] / vp["v_total"])


def _pappus(r, z):
    """Volume of revolution enclosed by a closed (R, Z) curve."""
    return abs(np.pi * np.sum(0.5 * (r[1:] ** 2 + r[:-1] ** 2) * np.diff(z)))
