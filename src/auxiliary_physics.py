"""Portable supplementary controls and graphical outer bounds."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
C0_MAX = 800.0
HENRY_K = 10.0
LDF_K = 0.1
Q = 1.0
AFFINITY = 1.0
BACKGROUND_A = (0.5, 1., 2.)
DOSES = (1., 2.)
A0_BOUNDS = (0.1, 2.)

class MonotoneRectangles:
    def __init__(self, frame):
        values = frame[["ce_low", "ce_high", "q_low", "q_high"]].to_numpy(float)
        if (not len(values) or not np.isfinite(values).all()
                or np.any(values[:, 0] < 0) or np.any(values[:, 0] > values[:, 1])
                or np.any(values[:, 2] > values[:, 3]) or np.any(values[:, 3] < 0)):
            raise ValueError("Invalid nonnegative uptake rectangles")
        self.xl, self.xh, self.yl, self.yh = values.T
        self.yl = np.maximum(self.yl, 0)
        self.nodes = np.unique(np.r_[0.0, self.xl, self.xh])
        for x in self.nodes:
            lo, hi = self.envelope(float(x))
            if lo > hi + 1e-10:
                raise ValueError("No continuous nondecreasing curve intersects all rectangles")

    def envelope(self, ce):
        if not np.isfinite(ce) or ce < 0:
            raise ValueError("Expected finite nonnegative concentration")
        lower = float(np.max(self.yl[self.xh <= ce], initial=0.0))
        upper = float(np.min(self.yh[self.xl >= ce], initial=np.inf))
        if ce == 0:
            upper = min(upper, 0.0)
        return lower, upper

    def root_bounds(self, c0, dose):
        if not np.isfinite([c0, dose]).all() or min(c0, dose) <= 0:
            raise ValueError("Expected positive finite C0 and dose")
        # Endpoint bounds include limits approached by steep continuous curves.
        lower = max(0.0, float(np.max(np.minimum(self.xl, c0 - dose*self.yh))))
        upper = min(c0, float(np.min(np.maximum(self.xh, c0 - dose*self.yl))))
        if lower > upper + 1e-10:
            raise AssertionError("Inconsistent mass-balance root bounds")
        return lower, upper

    def witnesses(self):
        bounds = np.array([self.envelope(float(x)) for x in self.nodes])
        cap = 2*max(1.0, float(self.yh.max()))
        low, high = bounds[:, 0], np.minimum(bounds[:, 1], cap)
        return [(self.nodes, (1-w)*low+w*high) for w in (0, 0.25, 0.5, 0.75, 1)]


def component_bounds(low, high, c0, low_dose, high_dose):
    if high_dose <= low_dose:
        raise ValueError("Expected ordered doses")
    ll, lu = low.root_bounds(c0, low_dose)
    hl, hu = high.root_bounds(c0, high_dose)
    high_ll, high_lu = high.envelope(ll), high.envelope(lu)
    low_hl, low_hu = low.envelope(hl), low.envelope(hu)
    a_low = high_ll[0] - (c0-ll)/low_dose
    a_high = high_lu[1] - (c0-lu)/low_dose
    b_low = (c0-hu)/high_dose - low_hu[1]
    b_high = (c0-hl)/high_dose - low_hl[0]
    depletion_low = 0.5*(low_hl[0]-(c0-ll)/low_dose
                         +(c0-hu)/high_dose-high_lu[1])
    depletion_high = 0.5*(low_hu[1]-(c0-lu)/low_dose
                          +(c0-hl)/high_dose-high_ll[0])
    if hu <= ll:
        depletion_high = min(depletion_high, 0.0)
    elif lu <= hl:
        depletion_low = max(depletion_low, 0.0)
    mixed_low = min(ll+high_dose*high_ll[0], hl+low_dose*low_hl[0])
    mixed_high = max(lu+high_dose*high_lu[1], hu+low_dose*low_hu[1])
    return dict(
        low_ce_lower=ll, low_ce_upper=lu, high_ce_lower=hl, high_ce_upper=hu,
        direct_lower=0.5*(a_low+b_low), direct_upper=0.5*(a_high+b_high),
        depletion_lower=depletion_low, depletion_upper=depletion_high,
        total_lower=(c0-hu)/high_dose-(c0-ll)/low_dose,
        total_upper=(c0-hl)/high_dose-(c0-lu)/low_dose,
        mixed_c0_lower=mixed_low, mixed_c0_upper=mixed_high,
        mixed_states_within_original_c0=bool(mixed_low >= 0 and mixed_high <= C0_MAX),
        all_high_dose_states_have_lower_ce=bool(hu <= ll),
        low_removal_lower=100*(1-lu/c0), low_removal_upper=100*(1-ll/c0),
        high_removal_lower=100*(1-hu/c0), high_removal_upper=100*(1-hl/c0),
    )


def linear_equilibrium(witness, c0, dose):
    x, y = witness
    if c0 > x[-1]:
        x, y = np.r_[x, c0], np.r_[y, y[-1]]
    balance = x+dose*y
    index = int(np.searchsorted(balance, c0, side="left"))
    if index == 0:
        return float(x[0])
    weight = (c0-balance[index-1])/(balance[index]-balance[index-1])
    return float(x[index-1]+weight*(x[index]-x[index-1]))


def linear_pair(low, high, c0, low_dose, high_dose):
    l = linear_equilibrium(low, c0, low_dose)
    h = linear_equilibrium(high, c0, high_dose)
    q00, q10 = float(np.interp(l, *low)), float(np.interp(l, *high))
    q01, q11 = float(np.interp(h, *low)), float(np.interp(h, *high))
    direct = 0.5*(q10-q00+q11-q01)
    depletion = 0.5*(q01-q00+q11-q10)
    total = q11-q00
    return dict(direct=direct, depletion=depletion, total=total, low_ce=l, high_ce=h,
                closure_error=abs(direct+depletion-total),
                balance_error=max(abs(l+low_dose*q00-c0), abs(h+high_dose*q11-c0)))


def bound_violation(bounds, row):
    values = [0.0]
    for name in ("direct", "depletion", "total", "low_ce", "high_ce"):
        values.extend([bounds[name+"_lower"]-row[name], row[name]-bounds[name+"_upper"]])
    return max(values)


def kinetic_state(c0, dose, time_h):
    c0, dose = np.broadcast_arrays(np.asarray(c0, float), np.asarray(dose, float))
    if (not np.isfinite(np.r_[c0.ravel(), dose.ravel(), time_h]).all()
            or np.any(c0 <= 0) or np.any(dose <= 0) or time_h < 0):
        raise ValueError("Expected positive coordinates and nonnegative finite time")
    decay = np.exp(-LDF_K * (1 + HENRY_K * dose) * time_h)
    q = HENRY_K * c0 / (1 + HENRY_K * dose) * (-np.expm1(
        -LDF_K * (1 + HENRY_K * dose) * time_h))
    ct = c0 * (1 + HENRY_K * dose * decay) / (1 + HENRY_K * dose)
    return dict(q=q, ct=ct, removal=100 * dose * q / c0,
                equilibrium_q=HENRY_K * c0 / (1 + HENRY_K * dose))


def apparent_slope(dose, time_h):
    dose = np.asarray(dose, float)
    state = kinetic_state(np.ones_like(dose), dose, time_h)
    return state["q"] / state["ct"]


def independent_ode(c0, dose, time_h):
    def rate(_, y):
        uptake = LDF_K * (HENRY_K * y[0] - y[1])
        return [-dose * uptake, uptake]
    solved = solve_ivp(rate, (0., time_h), (c0, 0.), method="Radau",
                       rtol=1e-10, atol=1e-11, t_eval=[time_h])
    if not solved.success:
        raise AssertionError(solved.message)
    return solved.y[:, -1]


@dataclass(frozen=True)
class State:
    a0_mmol_l: float
    b0_mmol_l: float
    dose_g_l: float
    a_mmol_l: float
    b_mmol_l: float
    qa_mmol_g: float
    qb_mmol_g: float
    free_site_fraction: float


def require_inputs(a0, b0, dose, capacity, ba, bb):
    values = np.asarray([a0, b0, dose, capacity, ba, bb], dtype=float)
    if not np.isfinite(values).all() or min(a0, b0) < 0:
        raise ValueError("Finite nonnegative initial concentrations required")
    if min(dose, capacity, ba, bb) <= 0:
        raise ValueError("Dose, capacity and affinities must be positive")


def equilibrate(a0, b0, dose, capacity=Q, ba=AFFINITY, bb=AFFINITY):
    require_inputs(a0, b0, dose, capacity, ba, bb)

    # The free-site equation has a positive derivative and a unique root.
    def balance(s):
        return (s + s * ba * a0 / (1 + dose * capacity * ba * s)
                + s * bb * b0 / (1 + dose * capacity * bb * s) - 1)

    free = brentq(balance, 0.0, 1.0, xtol=1e-14, rtol=1e-14)
    a = a0 / (1 + dose * capacity * ba * free)
    b = b0 / (1 + dose * capacity * bb * free)
    return State(a0, b0, dose, a, b, capacity * ba * a * free,
                 capacity * bb * b * free, free)


def equal_affinity_reference(a0, b0, dose, capacity=Q, affinity=AFFINITY):
    """Solve total residual concentration independently using a quadratic."""
    total = a0 + b0
    if total == 0:
        return (0.0, 0.0)
    coefficient = 1 + dose * capacity * affinity - total * affinity
    discriminant = np.hypot(coefficient, 2 * np.sqrt(affinity * total))
    if coefficient >= 0:
        residual = 2 * total / (discriminant + coefficient)
    else:
        residual = (discriminant - coefficient) / (2 * affinity)
    return residual * a0 / total, residual * b0 / total


def direct_at_shared_a(target, b0, dose):
    def balance(s):
        b = b0 / (1 + dose * Q * AFFINITY * s)
        return s * (1 + AFFINITY * target + AFFINITY * b) - 1

    free = brentq(balance, 0.0, 1.0, xtol=1e-14, rtol=1e-14)
    b = b0 / (1 + dose * Q * AFFINITY * free)
    qa = Q * AFFINITY * target * free
    qb = Q * AFFINITY * b * free
    return State(target + dose * qa, b0, dose, target, b, qa, qb, free)


def query_at_shared_a(predict, target, dose, bounds=A0_BOUNDS):
    """Use only predicted A uptake, not the competitive relation or B state."""
    lower, upper = bounds
    if not (np.isfinite([target, dose, lower, upper]).all()
            and target > 0 and dose > 0 and 0 < lower < upper):
        raise ValueError("Invalid positive target or query bounds")

    def residual(a0):
        return a0 - dose * float(predict(a0, dose)) - target

    left, right = residual(lower), residual(upper)
    if not np.isfinite([left, right]).all():
        return dict(status="nonfinite_endpoint", a0_mmol_l=np.nan)
    if left > 1e-12 or right < -1e-12:
        return dict(status="outside_initial_range", a0_mmol_l=np.nan)
    if abs(left) <= 1e-12:
        a0 = lower
    elif abs(right) <= 1e-12:
        a0 = upper
    else:
        a0 = brentq(residual, lower, upper, xtol=1e-13, rtol=1e-14)
    return dict(status="supported", a0_mmol_l=a0,
                qa_mmol_g=float(predict(a0, dose)),
                balance_residual_mmol_l=residual(a0))


def exact_shap(predict, a0, dose):
    baseline = float(np.mean([predict(a, d) for a in BACKGROUND_A for d in DOSES]))
    mean_a = float(np.mean([predict(a, dose) for a in BACKGROUND_A]))
    mean_d = float(np.mean([predict(a0, d) for d in DOSES]))
    output = float(predict(a0, dose))
    return dict(baseline_mmol_g=baseline,
                a0_shap_mmol_g=0.5 * (mean_d - baseline + output - mean_a),
                dose_shap_mmol_g=0.5 * (mean_a - baseline + output - mean_d))


