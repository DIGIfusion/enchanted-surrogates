"""
## Overview

The HelenaVolumeAverageSampler samples the *shape* of the Te and ne pedestal profiles
in **relative units** and then scales the profiles so that their **volume averages**
equal sampled target values. It emits absolute parameters named as in
`HelenaRunner` `input_parameter_type: 4` (no KBM constraint), see
`HelenaParser.write_input_file_noKBMconstraint`.

---
"""
import warnings

import numpy as np
import pandas as pd

from enchanted_surrogates.samplers.base_sampler import Sampler
from enchanted_surrogates.utils.volume_scaling import CURVE_COLUMNS, KNOTS, MEAN_V_CURVE, TANH_K, volume_weights

from enchanted_surrogates.utils.logger import get_logger
log = get_logger(__name__)

# Names kept for the ascot plugin (it compares its realised averages with this table): V(psiN)/V_total at psiN
_DEFAULT_PSI = KNOTS
_DEFAULT_V = MEAN_V_CURVE

_K = TANH_K

# Sampled parameters: name -> (default range, kind). kind "unit": must lie within [0, 1]; "pos": > 0.
# A parameter listed in `parameters` takes its range from the matching entry of `bounds`; the others keep the
# default here (None: not sampled). Te_avg / ne_avg have no default: give at least one of them.
_PARAMETERS = {
    # volume-average targets
    "Te_avg": (None, "pos"),                   # <Te> [keV]
    "ne_avg": (None, "pos"),                   # <ne> [1e19 m^-3]
    # relative shape, Te and ne
    "Te_ped_ov_core": ((0.01, 0.8), "unit"),   # T_ped / T_core
    "Te_sep_ov_ped": ((0.02, 0.3), "unit"),    # T_sep / T_ped
    "ne_ped_ov_core": ((0.01, 0.8), "unit"),   # n_ped / n_core
    "ne_sep_ov_ped": ((0.2, 0.5), "unit"),     # n_sep / n_ped
    # Ti/Te ratios (optional; ped and sep go together, core needs both)
    "Ti_ov_Te_ped": (None, "pos"),             # Ti(ped) / Te(ped)
    "Ti_ov_Te_sep": (None, "pos"),             # Ti(sep) / Te(sep)
    "Ti_ov_Te_core": (None, "pos"),            # Ti(0) / Te(0), axis values
    # pedestal widths in psiN
    "d_T_ped": ((0.01, 0.3), "pos"),
    "d_n_ped": ((0.01, 0.3), "pos"),
    # core shape exponents
    "alpha_T1": ((1.0, 2.5), "pos"),
    "alpha_T2": ((1.0, 2.5), "pos"),
    "alpha_n1": ((1.0, 2.5), "pos"),
    "alpha_n2": ((1.0, 2.5), "pos"),
}

# deprecated keyword -> parameter name(s)
_LEGACY = {
    "te_avg_range": ["Te_avg"], "ne_avg_range": ["ne_avg"],
    "te_ped_range": ["Te_ped_ov_core"], "te_sep_range": ["Te_sep_ov_ped"],
    "ne_ped_range": ["ne_ped_ov_core"], "ne_sep_range": ["ne_sep_ov_ped"],
    "ti_te_ped_range": ["Ti_ov_Te_ped"], "ti_te_sep_range": ["Ti_ov_Te_sep"], "ti_te_core_range": ["Ti_ov_Te_core"],
    "d_T_ped_range": ["d_T_ped"], "d_n_ped_range": ["d_n_ped"],
    "alpha_T_range": ["alpha_T1", "alpha_T2"], "alpha_n_range": ["alpha_n1", "alpha_n2"],
}


class HelenaVolumeAverageSampler(Sampler):
    """
    ## Configuration

    Parameters and ranges are set like in the other samplers: `parameters` names the sampled
    quantities and `bounds` gives each one's `[lo, hi]`, in the same order. A parameter that is
    not listed keeps its default range (or is off); `lo == hi` fixes a parameter.

    ```yaml
    sampler:
      type: HelenaVolumeAverageSampler
      budget: 100
      parameters: ['Te_avg', 'ne_avg', 'Te_ped_ov_core', 'Ti_ov_Te_ped', 'Ti_ov_Te_sep', 'Ti_ov_Te_core']
      bounds:
        - [3.1, 6.3]     # <Te> [keV]
        - [4.2, 8.1]     # <ne> [1e19 m^-3]
        - [0.3, 0.8]     # T_ped / T_core
        - [0.8, 1.2]     # Ti_ped / Te_ped
        - [1.0, 2.0]     # Ti_sep / Te_sep
        - [0.8, 1.2]     # Ti(0) / Te(0)
      volume_profile: null          # optional csv "psiN,V/V_total" (default: the mean curve, updated, see below)
      volume_prior_weight: 10       # weight of the default curve in equilibria when it is updated (0: do not update)
      seed: null
      batch_size: 1
      method: random                # or "sobol" (see below), or "design"
      design_csv: null              # only for method "design": csv with one column per sampled parameter, one row per sample
    ```

    With `method: design` the values are not drawn: row i of `design_csv` is sample i (e.g. one-at-a-time slices through the
    ranges). The columns are the parameters of this sampler (`Te_avg`, `ne_avg`, ... as in `parameters`), their values must lie
    in `bounds`, and `budget` must equal the number of rows. A row whose profile is not valid (core term < 0) raises an error
    instead of being redrawn, because a redraw would silently replace the design point by random values.

    ## Parameters

    All ranges are `[lo, hi]` and are drawn uniformly. Only the volume-average ranges
    come from machine tables; the other defaults are broad assumptions.

    | Parameter | Type / units | Default range | Meaning |
    |---|---|---|---|
    | `Te_avg` | keV | none | target volume-average Te; omit to skip Te |
    | `ne_avg` | 1e19 m^-3 | none | target volume-average ne; omit to skip ne |
    | `Te_ped_ov_core` | [0, 1] | [0.01, 0.8] | T_ped / T_core (core = psiN 0) |
    | `Te_sep_ov_ped` | [0, 1] | [0.02, 0.3] | T_sep / T_ped |
    | `ne_ped_ov_core` | [0, 1] | [0.01, 0.8] | n_ped / n_core |
    | `ne_sep_ov_ped` | [0, 1] | [0.2, 0.5] | n_sep / n_ped |
    | `Ti_ov_Te_ped` | > 0 | off | Ti_ped / Te_ped; give with `Ti_ov_Te_sep` |
    | `Ti_ov_Te_sep` | > 0 | off | Ti_sep / Te_sep; give with `Ti_ov_Te_ped` |
    | `Ti_ov_Te_core` | > 0 | off | Ti(0) / Te(0) on the magnetic axis; needs both ratios above. Draws whose implied Ti core term is negative are redrawn |
    | `d_T_ped` | psiN, > 0 | [0.01, 0.3] | Te pedestal width |
    | `d_n_ped` | psiN, > 0 | [0.01, 0.3] | ne pedestal width |
    | `alpha_T1`, `alpha_T2` | > 0 | [1, 2.5] | Te core exponents, drawn separately |
    | `alpha_n1`, `alpha_n2` | > 0 | [1, 2.5] | ne core exponents, drawn separately |

    Other options:

    | Option | Type | Default | Meaning |
    |---|---|---|---|
    | `budget` | int | required | total number of samples |
    | `batch_size` | int | 1 | samples per call |
    | `seed` | int | none | random seed (also the Sobol scrambling seed) |
    | `method` | `random` / `sobol` | `random` | `sobol`: every sample's drawn quantities come from one scrambled Sobol point (all species and the Ti ratios share the point). Use a power-of-2 `batch_size` (e.g. 256, drawn as one batch) to keep the Sobol balance properties |
    | `volume_profile` | csv path | mean curve | `psiN,V/V_total` table used for the volume average; a given table is not updated |
    | `volume_prior_weight` | equilibria | 10 | weight of the default curve when it is updated with solved equilibria; 0 keeps it fixed |

    The old keywords (`te_avg_range`, `ne_avg_range`, `te_ped_range`, `te_sep_range`, `ne_ped_range`,
    `ne_sep_range`, `ti_te_ped_range`, `ti_te_sep_range`, `ti_te_core_range`, `d_T_ped_range`,
    `d_n_ped_range`, `alpha_T_range`, `alpha_n_range`) still work but are deprecated; they cannot be
    mixed with `parameters`/`bounds`.

    At least one of `Te_avg` / `ne_avg` is required; the Ti ratios need Te
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
    | `Ti_ov_Te_core` | - | Ti(0)/Te(0) on the axis, only if configured; the parser turns it into the Ti core term `cti` |

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

    - Without `Ti_ov_Te_core` the parser does not set the Ti core term: with Ti ratios
      the Ti pedestal/separatrix are set, and the Ti core is whatever the base namelist's
      Ti core term gives (0: Ti flat to the pedestal top). The Ti volume average is not scaled here.
    - With `Ti_ov_Te_core` the Ti axis value is Ti_ov_Te_core * Te(0), where Te(0) is the
      scale factor of the Te profile. The Ti core term is then
      `Ti(0) - Ti_sep - ati*c`, `c = tanh(1) + tanh(2(1-delta/2)/delta)`, and must be >= 0.
    - `density_shift` is not sampled (0).
    - Only the volume-average ranges come from machine tables; the relative-parameter,
      width and exponent defaults are broad assumptions, not device data.
    - The volume average uses V(psiN)/V_total, the mean curve of 115 equilibria of a first JT-60SA run
      (`utils.volume_scaling.MEAN_V_CURVE`) by default. It is updated while the run goes on: after every batch
      the curves of the successful equilibria (the `V_over_Vtot_*` columns that `HelenaRunner` returns) are averaged
      into it, so that the later samples are scaled with a curve closer to the equilibria that this run produces. This needs
      batches (`batch_size` smaller than the budget) and a join sampler that passes the results on. The samples
      also carry the targets, `Te_avg_target` and `ne_avg_target`, which `HelenaRunner` uses to correct the scaling
      once the equilibrium is solved (`volume_scaling_iterations`). Supply
      `volume_profile` for another device.

    ---
    """

    MAX_REDRAWS = 1000

    def __init__(self, budget, parameters=None, bounds=None, volume_profile=None,
                 seed=None, batch_size=1, method="random", volume_prior_weight=10.0, design_csv=None, **kwargs):
        self.budget = budget
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        if method not in ("random", "sobol", "design"):
            raise ValueError(f"method must be 'random', 'sobol' or 'design', got {method!r}")
        self.method = method
        self._unit = None      # iterator over the unit-cube numbers of the current sample

        # Ranges of the sampled parameters; `parameters`/`bounds` (or the deprecated *_range keywords)
        given = self._read_ranges(parameters, bounds, kwargs)
        self.opt = {}
        for name, (default, kind) in _PARAMETERS.items():
            value = given.get(name, default)
            self.opt[name] = None if value is None else self._check(name, value, kind)
        self.te_avg, self.ne_avg = self.opt["Te_avg"], self.opt["ne_avg"]
        if self.te_avg is None and self.ne_avg is None:
            raise ValueError("give Te_avg and/or ne_avg in parameters/bounds")

        ti_keys = [self.opt["Ti_ov_Te_ped"], self.opt["Ti_ov_Te_sep"]]
        self._ti = any(k is not None for k in ti_keys)
        if self._ti and not all(k is not None for k in ti_keys):
            raise ValueError("give both Ti_ov_Te_ped and Ti_ov_Te_sep (or neither)")
        if self._ti and self.te_avg is None:
            raise ValueError("the Ti ratios need Te_avg")
        self._ti_core = self.opt["Ti_ov_Te_core"] is not None
        if self._ti_core and not self._ti:
            raise ValueError("Ti_ov_Te_core needs Ti_ov_Te_ped and Ti_ov_Te_sep")

        # Names of the parameters in each sample (needed e.g. by SampleWiseJoinSampler)
        self.parameters = []
        if self.te_avg:
            self.parameters += ["T_eped", "T_esep", "aT1", "d_T_ped", "alpha_T1", "alpha_T2", "Te_avg_target"]
        if self.ne_avg:
            self.parameters += ["n_eped", "n_esep", "an1", "d_n_ped", "alpha_n1", "alpha_n2", "ne_avg_target"]
        if self._ti:
            self.parameters += ["Ti_ov_Te_ped", "Ti_ov_Te_sep"]
        if self._ti_core:
            self.parameters += ["Ti_ov_Te_core"]

        self._design = None
        if method == "design":
            self._design = self._read_design(design_csv)
            if len(self._design) != budget:
                raise ValueError(f"budget is {budget} but the design has {len(self._design)} rows")
            self._design_row = 0
        if method == "sobol":
            from scipy.stats import qmc
            # per species: avg, ped, sep, width, alpha1, alpha2; Ti: two ratios
            dim = (6 * (self.te_avg is not None) + 6 * (self.ne_avg is not None) + 2 * self._ti
                   + 1 * self._ti_core)
            self._sobol = qmc.Sobol(d=dim, scramble=True, seed=seed)

        if volume_profile is None:
            prior = MEAN_V_CURVE
            # the prior is updated with the solved equilibria that come back, see register_future
            self._update_curve = volume_prior_weight is not None and volume_prior_weight > 0
        else:
            data = np.loadtxt(volume_profile, delimiter=",", comments="#")
            psi_v, v = data[:, 0], data[:, 1] / data[-1, 1]
            prior = np.interp(KNOTS, np.r_[0.0, psi_v] if psi_v[0] > 0 else psi_v, np.r_[0.0, v] if psi_v[0] > 0 else v)
            self._update_curve = False        # a table given for another device stays as it is
        self._curve_sum = float(volume_prior_weight or 0.0) * np.asarray(prior, dtype=float)
        self._curve_weight = float(volume_prior_weight or 0.0)
        self._set_curve(np.asarray(prior, dtype=float))

    def _set_curve(self, curve):
        """Use `curve` (V/V_total at KNOTS) as the assumed V(psiN) from now on."""
        self.curve = curve
        self._psi, self._dv = volume_weights(curve, 800)

    @staticmethod
    def _read_ranges(parameters, bounds, legacy_kwargs):
        """{parameter name: [lo, hi]} from parameters/bounds, or from the deprecated *_range keywords."""
        legacy = {k: v for k, v in legacy_kwargs.items() if k in _LEGACY}
        unknown = set(legacy_kwargs) - set(legacy)
        if unknown:
            raise ValueError(f"unknown sampler options: {sorted(unknown)}")
        if legacy and (parameters is not None or bounds is not None):
            raise ValueError("do not mix parameters/bounds with the deprecated *_range keywords")
        if legacy:
            warnings.warn("The *_range keywords of HelenaVolumeAverageSampler are deprecated; use parameters "
                          "and bounds.", DeprecationWarning, stacklevel=3)
            return {name: v for k, v in legacy.items() for name in _LEGACY[k]}
        parameters, bounds = list(parameters or []), list(bounds or [])
        if len(parameters) != len(bounds):
            raise ValueError(f"parameters ({len(parameters)}) and bounds ({len(bounds)}) must have the same length")
        if len(set(parameters)) != len(parameters):
            raise ValueError(f"duplicate names in parameters: {parameters}")
        bad = [p for p in parameters if p not in _PARAMETERS]
        if bad:
            raise ValueError(f"unknown parameters {bad}; valid names: {list(_PARAMETERS)}")
        return dict(zip(parameters, bounds))

    def _design_names(self):
        """Parameters in the order in which `_draw` and `_draw_ti` take their uniform numbers."""
        names = []
        if self.te_avg:
            names += ["Te_avg", "Te_ped_ov_core", "Te_sep_ov_ped", "d_T_ped", "alpha_T1", "alpha_T2"]
        if self.ne_avg:
            names += ["ne_avg", "ne_ped_ov_core", "ne_sep_ov_ped", "d_n_ped", "alpha_n1", "alpha_n2"]
        if self._ti:
            names += ["Ti_ov_Te_ped", "Ti_ov_Te_sep"]
        if self._ti_core:
            names += ["Ti_ov_Te_core"]
        return names

    def _read_design(self, path):
        """Unit-cube coordinates (rows x dims, in `_design_names` order) of the design in `path`."""
        if path is None:
            raise ValueError("method 'design' needs design_csv")
        table = np.genfromtxt(path, delimiter=",", names=True)
        table = np.atleast_1d(table)
        names = self._design_names()
        missing = [n for n in names if n not in table.dtype.names]
        if missing:
            raise ValueError(f"design_csv {path} has no column(s) {missing}")
        unit = np.empty((len(table), len(names)))
        for j, name in enumerate(names):
            lo, hi = self.opt[name]
            v = table[name]
            if np.any(v < lo - 1e-12 * max(1.0, abs(lo))) or np.any(v > hi + 1e-12 * max(1.0, abs(hi))):
                raise ValueError(f"design column {name} has values outside its bounds {[lo, hi]}")
            unit[:, j] = 0.0 if hi == lo else (v - lo) / (hi - lo)
        return np.clip(unit, 0.0, 1.0)

    def _u(self, lo, hi):
        """Uniform draw: next Sobol coordinate of the current sample, else pseudo-random."""
        if self._unit is not None:
            try:
                return lo + (hi - lo) * next(self._unit)
            except StopIteration:
                if self._design is not None:
                    raise RuntimeError("the design point ran out of numbers: a redraw would replace it by random values")
                # redraws after a rejected draw fall back to pseudo-random
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

    def _draw(self, avg_range, ped_r, sep_r, width_r, alpha1_r, alpha2_r):
        u = self._u
        for _ in range(self.MAX_REDRAWS):
            avg = u(*avg_range)
            ped_frac, sep_rel = u(*ped_r), u(*sep_r)
            delta = u(*width_r)
            alpha1, alpha2 = u(*alpha1_r), u(*alpha2_r)
            prof = self._unit_profile(ped_frac, sep_rel, delta, alpha1, alpha2)
            if prof is None:
                if self._design is not None:
                    raise ValueError(f"design point: the profile with pedestal-top fraction {ped_frac:.4g}, "
                                     f"separatrix ratio {sep_rel:.4g}, width {delta:.4g} has a negative core term")
                continue
            a0, sep, a1, t1, t2 = prof
            unit_avg = np.sum((sep + a0 * t1 + a1 * t2) * self._dv)
            scale = avg / unit_avg                    # exact: profile is linear in scale
            return dict(ped=ped_frac * scale, sep=sep * scale, core=a1 * scale, axis=scale, avg=avg,
                        delta=delta, alpha1=alpha1, alpha2=alpha2)
        raise RuntimeError(
            f"Could not draw a valid profile in {self.MAX_REDRAWS} tries; the pedestal-top "
            "fraction is probably too close to 1 for the chosen sep ratio and width."
        )

    def _draw_ti(self, te):
        """Ti/Te ratios at the pedestal top, separatrix and (optionally) axis.

        The Ti profile uses the Te width. Its core term is Ti(0) - Ti_sep - ati*c and is required to be >= 0
        (no hollow core), like the Te core term; draws that violate it are redrawn."""
        o = self.opt
        c = np.tanh(1.0) + np.tanh(2 * (1 - te["delta"] / 2) / te["delta"])
        for _ in range(self.MAX_REDRAWS):
            r_ped, r_sep = float(self._u(*o["Ti_ov_Te_ped"])), float(self._u(*o["Ti_ov_Te_sep"]))
            out = {"Ti_ov_Te_ped": r_ped, "Ti_ov_Te_sep": r_sep}
            if not self._ti_core:
                return out
            r_core = float(self._u(*o["Ti_ov_Te_core"]))
            ti_ped, ti_sep = r_ped * te["ped"], r_sep * te["sep"]
            core_term = r_core * te["axis"] - ti_sep - (ti_ped - ti_sep) / _K * c
            if core_term >= 0:
                out["Ti_ov_Te_core"] = r_core
                return out
            if self._design is not None:
                raise ValueError(f"design point with Ti_ov_Te_core {r_core:.4g} has a negative Ti core term")
        raise RuntimeError(
            f"Could not draw Ti ratios with a non-negative Ti core term in {self.MAX_REDRAWS} tries; "
            "widen the Ti_ov_Te_core bounds upward or lower the pedestal/separatrix ratios."
        )

    def get_next_samples(self) -> list[dict]:
        o = self.opt
        samples = []
        unit = self._sobol.random(self.batch_size) if self.method == "sobol" else None
        n = self.batch_size
        if self._design is not None:
            unit = self._design[self._design_row:self._design_row + self.batch_size]
            self._design_row += len(unit)
            n = len(unit)
        for i in range(n):
            self._unit = iter(unit[i]) if unit is not None else None
            sample = {}
            if self.te_avg:
                te = self._draw(self.te_avg, o["Te_ped_ov_core"], o["Te_sep_ov_ped"],
                                o["d_T_ped"], o["alpha_T1"], o["alpha_T2"])
                sample.update(T_eped=te["ped"], T_esep=te["sep"], aT1=te["core"],
                              d_T_ped=te["delta"], alpha_T1=te["alpha1"], alpha_T2=te["alpha2"],
                              Te_avg_target=te["avg"])
            if self.ne_avg:
                ne = self._draw(self.ne_avg, o["ne_ped_ov_core"], o["ne_sep_ov_ped"],
                                o["d_n_ped"], o["alpha_n1"], o["alpha_n2"])
                sample.update(n_eped=ne["ped"], n_esep=ne["sep"], an1=ne["core"],
                              d_n_ped=ne["delta"], alpha_n1=ne["alpha1"], alpha_n2=ne["alpha2"],
                              ne_avg_target=ne["avg"])
            if self._ti:
                sample.update(self._draw_ti(te))
            samples.append({k: float(v) for k, v in sample.items()})
        self._unit = None
        self.submitted += len(samples)
        return samples

    def register_future(self, future):
        """Update the assumed V(psiN)/V_total with the solved equilibria of a finished batch.

        `future` is the dataset of the batch. Its successful rows that carry the V(psiN)/V_total curve of their equilibrium
        (the V_over_Vtot_* columns that HelenaRunner returns) are averaged into the curve, which starts as the table (the
        mean of an earlier run) with the weight `volume_prior_weight`, in equilibria. The next samples are scaled with the
        updated curve, so that their volume averages are closer to the targets before the runner has to correct them.
        """
        if not self._update_curve or not isinstance(future, pd.DataFrame) or future.empty:
            return None
        if not set(CURVE_COLUMNS) <= set(future.columns):
            return None
        rows = future
        if "success" in rows:
            rows = rows[rows["success"] == True]   # noqa: E712
        curves = rows[CURVE_COLUMNS].dropna().to_numpy(dtype=float)
        if len(curves) == 0:
            return None
        self._curve_sum = self._curve_sum + curves.sum(axis=0)
        self._curve_weight += len(curves)
        self._set_curve(self._curve_sum / self._curve_weight)
        log.info(f"Volume curve updated with {len(curves)} equilibria (weight {self._curve_weight:g}): "
                 f"V/V_total at psiN = 0.5 is {self.curve[10]:.3f}")
        return None

    def register_futures(self, futures):
        return None
