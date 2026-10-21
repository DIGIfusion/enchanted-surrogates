"""
## Overview

The HelenaVolumeAverageSampler samples the *shape* of the Te and ne pedestal profiles
in **relative units** and then scales the profiles so that their **volume averages**
equal sampled target values. It emits absolute parameters named as in
`HelenaRunner` `input_parameter_type: 4` (no KBM constraint), see
`HelenaParser.write_input_file_noKBMconstraint`.

---
"""
import numpy as np

from enchanted_surrogates.samplers.base_sampler import Sampler

from enchanted_surrogates.utils.logger import get_logger
log = get_logger(__name__)

# V(psiN)/V_total from a HELENA run of JT-60SA scenario #2 (Table 1-1 geometry).
# Used for the volume average; override with `volume_profile`.
_DEFAULT_PSI = [0.0, 0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3, 0.34, 0.38, 0.42, 0.46,
                0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82, 0.86, 0.9, 0.92, 0.94,
                0.96, 0.97, 0.98, 0.99, 1.0]
_DEFAULT_V = [0.0, 0.00728, 0.02243, 0.03844, 0.05543, 0.0735, 0.0927, 0.11313, 0.13485,
              0.15796, 0.18255, 0.20879, 0.23675, 0.26663, 0.29866, 0.333, 0.36989, 0.40978,
              0.45303, 0.50013, 0.5518, 0.60897, 0.67301, 0.74607, 0.78702, 0.8317, 0.88083,
              0.90733, 0.93549, 0.96594, 1.0]

# Same constant as helena_parser.TANH_NORMALIZER = tanh(1) - tanh(-1)
_K = 2 * np.tanh(1.0)

# name -> (default range, kind). kind "unit": must lie within [0, 1]; "pos": > 0.
_SHAPE_OPTIONS = {
    # relative units
    "te_ped_range": ((0.01, 0.8), "unit"),    # T_ped / T_core
    "te_sep_range": ((0.02, 0.3), "unit"),    # T_sep / T_ped
    "ne_ped_range": ((0.01, 0.8), "unit"),    # n_ped / n_core
    "ne_sep_range": ((0.2, 0.5), "unit"),     # n_sep / n_ped
    "ti_te_ped_range": (None, "pos"),         # Ti(ped) / Te(ped)   (optional)
    "ti_te_sep_range": (None, "pos"),         # Ti(sep) / Te(sep)   (optional)
    # pedestal widths in psiN
    "d_T_ped_range": ((0.01, 0.3), "pos"),
    "d_n_ped_range": ((0.01, 0.3), "pos"),
    # core shape exponents
    "alpha_T_range": ((1.0, 2.5), "pos"),
    "alpha_n_range": ((1.0, 2.5), "pos"),
}


class HelenaVolumeAverageSampler(Sampler):
    """
    ## Configuration

    ```yaml
    sampler:
      type: HelenaVolumeAverageSampler
      budget: 100
      # volume-average targets, sampled uniformly (omit one to skip that species)
      te_avg_range: [3.1, 6.3]      # <Te> [keV]
      ne_avg_range: [4.2, 8.1]      # <ne> [1e19 m^-3]
      # relative shape parameters (all optional, defaults shown)
      te_ped_range: [0.01, 0.8]     # T_ped / T_core        (core = value at psiN=0)
      te_sep_range: [0.02, 0.3]     # T_sep / T_ped
      ne_ped_range: [0.01, 0.8]     # n_ped / n_core
      ne_sep_range: [0.2, 0.5]      # n_sep / n_ped
      ti_te_ped_range: [0.8, 1.2]   # Ti_ped / Te_ped   (optional, needs Te)
      ti_te_sep_range: [1.0, 3.0]   # Ti_sep / Te_sep   (optional, needs Te)
      d_T_ped_range: [0.01, 0.3]    # pedestal widths [psiN]
      d_n_ped_range: [0.01, 0.3]
      alpha_T_range: [1.0, 2.5]     # core exponents (alpha_T1, alpha_T2)
      alpha_n_range: [1.0, 2.5]     # core exponents (alpha_n1, alpha_n2)
      volume_profile: null          # optional csv "psiN,V/V_total" (default: JT-60SA #2)
      seed: null
      batch_size: 1
      method: random                # or "sobol" (see below)
    ```

    ## Input options

    All ranges are `[lo, hi]` and are drawn uniformly. Only the volume-average ranges
    come from machine tables; the other defaults are broad assumptions.

    | Option | Type / units | Default | Meaning |
    |---|---|---|---|
    | `budget` | int | required | total number of samples |
    | `batch_size` | int | 1 | samples per call |
    | `seed` | int | none | random seed (also the Sobol scrambling seed) |
    | `method` | `random` / `sobol` | `random` | `sobol`: every sample's drawn quantities come from one scrambled Sobol point (all species and the Ti ratios share the point). Use a power-of-2 `batch_size` (e.g. 256, drawn as one batch) to keep the Sobol balance properties |
    | `volume_profile` | csv path | JT-60SA #2 | `psiN,V/V_total` table used for the volume average |
    | `te_avg_range` | keV | none | target volume-average Te; omit to skip Te |
    | `ne_avg_range` | 1e19 m^-3 | none | target volume-average ne; omit to skip ne |
    | `te_ped_range` | [0, 1] | [0.01, 0.8] | T_ped / T_core (core = psiN 0) |
    | `te_sep_range` | [0, 1] | [0.02, 0.3] | T_sep / T_ped |
    | `ne_ped_range` | [0, 1] | [0.01, 0.8] | n_ped / n_core |
    | `ne_sep_range` | [0, 1] | [0.2, 0.5] | n_sep / n_ped |
    | `ti_te_ped_range` | > 0 | off | Ti_ped / Te_ped; give with `ti_te_sep_range` |
    | `ti_te_sep_range` | > 0 | off | Ti_sep / Te_sep; give with `ti_te_ped_range` |
    | `d_T_ped_range` | psiN, > 0 | [0.01, 0.3] | Te pedestal width |
    | `d_n_ped_range` | psiN, > 0 | [0.01, 0.3] | ne pedestal width |
    | `alpha_T_range` | > 0 | [1, 2.5] | Te core exponents (`alpha_T1`, `alpha_T2`, drawn separately) |
    | `alpha_n_range` | > 0 | [1, 2.5] | ne core exponents (`alpha_n1`, `alpha_n2`, drawn separately) |

    At least one of `te_avg_range` / `ne_avg_range` is required; the Ti ratios need Te
    to be sampled too. `ped` is the profile value at psiN = 1 - width (pedestal top),
    and a `sep` ratio is relative to that pedestal-top value, so the separatrix can
    never exceed it.

    ## Output

    Each sample is a dict of absolute values, with the names read by
    `HelenaParser.write_input_file_noKBMconstraint` (use `input_parameter_type: 4`):

    | Output | Units | Meaning |
    |---|---|---|
    | `T_eped` | keV | Te at the pedestal top |
    | `T_esep` | keV | Te at the separatrix |
    | `aT1` | keV | Te core term (`cte`); set by the shape and the volume average, not drawn |
    | `d_T_ped` | psiN | Te pedestal width |
    | `alpha_T1`, `alpha_T2` | - | Te core exponents |
    | `n_eped`, `n_esep` | 1e19 m^-3 | ne at the pedestal top and separatrix |
    | `an1` | 1e19 m^-3 | ne core term (`cde`); set by the shape and the volume average, not drawn |
    | `d_n_ped` | psiN | ne pedestal width |
    | `alpha_n1`, `alpha_n2` | - | ne core exponents |
    | `Ti_ov_Te_ped`, `Ti_ov_Te_sep` | - | Ti/Te ratios, only if configured |

    The exponents, widths and relative ratios are dimensionless and fixed by the
    sampled shape; the volume-average scaling changes only the absolute outputs
    (`T_eped`, `T_esep`, `aT1`, `n_eped`, `n_esep`, `an1`).

    The core term `a1` is `1 - sep - a0*C` for a unit core value, so it depends only on
    the pedestal-top fraction, the separatrix ratio and the width, and the absolute
    value is that times the volume-average scale factor. The exponents affect it only
    through that scale factor.

    ## Model

    With `mid = 1 - delta/2` and `ped = 1 - delta`:

        f = sep + a0*[tanh(2(1-mid)/delta) - tanh(2(psiN-mid)/delta)]
                + a1*(1-(psiN/ped)^alpha1)^alpha2   (psiN <= ped)

    `T_eped`/`n_eped` is f at psiN = ped and a0 = (f_ped - sep)/_K (the parser's
    convention). Per sample and species: the relative parameters, width and exponents
    are drawn uniformly; a unit-core profile is built from them; and it is multiplied
    by the factor that makes its volume average equal the drawn target. The core term
    `a1` then follows from the shape (it is not sampled separately). Draws with
    `a1 < 0` (a pedestal-top fraction too close to 1 for the chosen sep and width)
    are redrawn.

    ## Notes

    - The parser does not take the Ti core term from this sampler: with Ti ratios
      the Ti pedestal/separatrix are set, and the Ti volume average is whatever the
      base namelist's Ti core term gives. It is not scaled here.
    - `density_shift` is not sampled (0).
    - Only the volume-average ranges come from machine tables; the relative-parameter,
      width and exponent defaults are broad assumptions, not device data.
    - The volume average uses V(psiN) of JT-60SA scenario #2 by default; supply
      `volume_profile` for another device.

    ---
    """

    MAX_REDRAWS = 1000

    def __init__(self, budget, te_avg_range=None, ne_avg_range=None, volume_profile=None,
                 seed=None, batch_size=1, method="random", **kwargs):
        self.budget = budget
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        if method not in ("random", "sobol"):
            raise ValueError(f"method must be 'random' or 'sobol', got {method!r}")
        self.method = method
        self._unit = None      # iterator over the unit-cube numbers of the current sample

        unknown = set(kwargs) - set(_SHAPE_OPTIONS)
        if unknown:
            raise ValueError(f"unknown sampler options: {sorted(unknown)}")
        if te_avg_range is None and ne_avg_range is None:
            raise ValueError("give te_avg_range and/or ne_avg_range")
        self.te_avg = self._check("te_avg_range", te_avg_range, "pos") if te_avg_range else None
        self.ne_avg = self._check("ne_avg_range", ne_avg_range, "pos") if ne_avg_range else None

        self.opt = {}
        for name, (default, kind) in _SHAPE_OPTIONS.items():
            value = kwargs.get(name, default)
            self.opt[name] = None if value is None else self._check(name, value, kind)

        ti_keys = [self.opt["ti_te_ped_range"], self.opt["ti_te_sep_range"]]
        self._ti = any(k is not None for k in ti_keys)
        if self._ti and not all(k is not None for k in ti_keys):
            raise ValueError("give both ti_te_ped_range and ti_te_sep_range (or neither)")
        if self._ti and self.te_avg is None:
            raise ValueError("ti_te_*_range needs te_avg_range")

        # Names of the parameters in each sample (needed e.g. by SampleWiseJoinSampler)
        self.parameters = []
        if self.te_avg:
            self.parameters += ["T_eped", "T_esep", "aT1", "d_T_ped", "alpha_T1", "alpha_T2"]
        if self.ne_avg:
            self.parameters += ["n_eped", "n_esep", "an1", "d_n_ped", "alpha_n1", "alpha_n2"]
        if self._ti:
            self.parameters += ["Ti_ov_Te_ped", "Ti_ov_Te_sep"]

        if method == "sobol":
            from scipy.stats import qmc
            # per species: avg, ped, sep, width, alpha1, alpha2; Ti: two ratios
            dim = 6 * (self.te_avg is not None) + 6 * (self.ne_avg is not None) + 2 * self._ti
            self._sobol = qmc.Sobol(d=dim, scramble=True, seed=seed)

        if volume_profile is None:
            psi_v, v = np.array(_DEFAULT_PSI), np.array(_DEFAULT_V)
        else:
            data = np.loadtxt(volume_profile, delimiter=",", comments="#")
            psi_v, v = data[:, 0], data[:, 1]
            v = v / v[-1]
        grid = np.linspace(0.0, 1.0, 801)
        vcum = np.interp(grid, np.r_[0.0, psi_v] if psi_v[0] > 0 else psi_v,
                         np.r_[0.0, v] if psi_v[0] > 0 else v)
        dv = np.diff(vcum)
        self._dv = dv / dv.sum()
        self._psi = 0.5 * (grid[1:] + grid[:-1])

    def _u(self, lo, hi):
        """Uniform draw: next Sobol coordinate of the current sample, else pseudo-random."""
        if self._unit is not None:
            try:
                return lo + (hi - lo) * next(self._unit)
            except StopIteration:
                pass   # redraws after a rejected draw fall back to pseudo-random
        return self.rng.uniform(lo, hi)

    @staticmethod
    def _check(name, value, kind):
        lo, hi = (float(x) for x in value)
        if lo > hi:
            raise ValueError(f"{name}: lower bound {lo} > upper bound {hi}")
        if kind == "unit" and not (0.0 <= lo and hi <= 1.0):
            raise ValueError(f"{name} must lie within [0, 1], got {[lo, hi]}")
        if kind == "pos" and lo <= 0:
            raise ValueError(f"{name} must be > 0, got {[lo, hi]}")
        return (lo, hi)

    def _unit_profile(self, ped_frac, sep_over_ped, delta, alpha1, alpha2):
        """
        Profile with core value (psiN = 0) of 1, from the relative parameters.
        Returns (a0, sep, a1, mid, ped, t1, t2) or None if the core term a1 < 0.
        """
        sep = ped_frac * sep_over_ped
        a0 = (ped_frac - sep) / _K
        c = np.tanh(1.0) + np.tanh(2 * (1 - delta / 2) / delta)
        a1 = 1.0 - sep - a0 * c
        if a1 < 0:
            return None
        mid, ped = 1 - delta / 2, 1 - delta
        psi = self._psi
        t1 = np.tanh(2 * (1 - mid) / delta) - np.tanh(2 * (psi - mid) / delta)
        x = np.clip(psi / ped, 0, 1)
        t2 = np.where(psi <= ped, (1 - x ** alpha1) ** alpha2, 0.0)
        return a0, sep, a1, t1, t2

    def _draw(self, avg_range, ped_r, sep_r, width_r, alpha_r):
        u = self._u
        for _ in range(self.MAX_REDRAWS):
            avg = u(*avg_range)
            ped_frac, sep_rel = u(*ped_r), u(*sep_r)
            delta = u(*width_r)
            alpha1, alpha2 = u(*alpha_r), u(*alpha_r)
            prof = self._unit_profile(ped_frac, sep_rel, delta, alpha1, alpha2)
            if prof is None:
                continue
            a0, sep, a1, t1, t2 = prof
            unit_avg = np.sum((sep + a0 * t1 + a1 * t2) * self._dv)
            scale = avg / unit_avg                    # exact: profile is linear in scale
            return dict(ped=ped_frac * scale, sep=sep * scale, core=a1 * scale,
                        delta=delta, alpha1=alpha1, alpha2=alpha2)
        raise RuntimeError(
            f"Could not draw a valid profile in {self.MAX_REDRAWS} tries; the pedestal-top "
            "fraction is probably too close to 1 for the chosen sep ratio and width."
        )

    def get_next_samples(self) -> list[dict]:
        o = self.opt
        samples = []
        unit = self._sobol.random(self.batch_size) if self.method == "sobol" else None
        for i in range(self.batch_size):
            self._unit = iter(unit[i]) if unit is not None else None
            sample = {}
            if self.te_avg:
                te = self._draw(self.te_avg, o["te_ped_range"], o["te_sep_range"],
                                o["d_T_ped_range"], o["alpha_T_range"])
                sample.update(T_eped=te["ped"], T_esep=te["sep"], aT1=te["core"],
                              d_T_ped=te["delta"], alpha_T1=te["alpha1"], alpha_T2=te["alpha2"])
            if self.ne_avg:
                ne = self._draw(self.ne_avg, o["ne_ped_range"], o["ne_sep_range"],
                                o["d_n_ped_range"], o["alpha_n_range"])
                sample.update(n_eped=ne["ped"], n_esep=ne["sep"], an1=ne["core"],
                              d_n_ped=ne["delta"], alpha_n1=ne["alpha1"], alpha_n2=ne["alpha2"])
            if self._ti:
                sample["Ti_ov_Te_ped"] = float(self._u(*o["ti_te_ped_range"]))
                sample["Ti_ov_Te_sep"] = float(self._u(*o["ti_te_sep_range"]))
            samples.append({k: float(v) for k, v in sample.items()})
        self._unit = None
        self.submitted += len(samples)
        return samples

    def register_future(self, future):
        return None

    def register_futures(self, futures):
        return None
