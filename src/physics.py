"""Physical references, fixed-background games, and recorded-condition comparisons."""

from math import factorial
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq

DOSE = "Dose (g/L)"
UPTAKE = "Adsorption capacity (mg/g)"
C0 = "Initial concentration (mg/L)"
SCAN_POINTS = 513
QMAX = 100.0
K = 0.1

def checked_inputs(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or len(values) == 0
            or not np.isfinite(values).all() or np.any(values <= 0)):
        raise ValueError("Expected nonempty finite positive (C0, V, m) rows")
    return values


def ceiling(values):
    values = checked_inputs(values)
    return values[:, 0] * values[:, 1] / values[:, 2]


def coalition_values(background, queries):
    """Joint empirical marginalization, with all missing inputs from one row."""
    background, queries = checked_inputs(background), checked_inputs(queries)
    c0, volume, mass = background.T
    x_c0, x_volume, x_mass = queries.T
    values = np.empty((len(queries), 8), dtype=float)
    values[:, 0] = np.mean(c0 * volume / mass)
    values[:, 1] = x_c0 * np.mean(volume / mass)
    values[:, 2] = x_volume * np.mean(c0 / mass)
    values[:, 3] = x_c0 * x_volume * np.mean(1 / mass)
    values[:, 4] = np.mean(c0 * volume) / x_mass
    values[:, 5] = x_c0 * np.mean(volume) / x_mass
    values[:, 6] = x_volume * np.mean(c0) / x_mass
    values[:, 7] = ceiling(queries)
    if not np.isfinite(values).all():
        raise ValueError("The ceiling game overflowed")
    return values


def exact_attributions(background, queries):
    values = coalition_values(background, queries)
    phi = np.zeros((len(values), 3), dtype=float)
    for player in range(3):
        for mask in range(8):
            if mask & (1 << player):
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(2 - size) / factorial(3)
            phi[:, player] += weight * (values[:, mask | (1 << player)] - values[:, mask])
    return phi, values


def grouped_contributions(grid, joint_weights, context_ids, mass_ids):
    """Two players: mass and the joint block of all remaining inputs."""
    grid = np.asarray(grid, float)
    weights = np.asarray(joint_weights, float)
    context_ids, mass_ids = np.asarray(context_ids), np.asarray(mass_ids)
    if (grid.ndim != 2 or weights.shape != grid.shape
            or not np.isfinite(grid).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1)):
        raise ValueError("Expected a finite grid and normalized background weights")
    if context_ids.shape != mass_ids.shape or context_ids.ndim != 1:
        raise ValueError("Expected paired query indices")
    baseline = float(np.sum(grid * weights))
    context_coalition = grid[context_ids] @ weights.sum(axis=0)
    mass_coalition = weights.sum(axis=1) @ grid[:, mass_ids]
    predicted = grid[context_ids, mass_ids]
    mass_phi = (mass_coalition - baseline + predicted - context_coalition) / 2
    context_phi = (context_coalition - baseline + predicted - mass_coalition) / 2
    return dict(predicted=predicted, baseline=baseline, mass_phi=mass_phi,
                context_phi=context_phi, context_coalition=context_coalition,
                mass_coalition=mass_coalition)


def removal_grid(q_grid, contexts, masses):
    masses = np.asarray(masses, float)
    solute_mass = contexts.Ci.to_numpy(float) * contexts["Volume (L)"].to_numpy(float)
    if np.any(solute_mass <= 0) or np.any(masses <= 0):
        raise ValueError("Initial solute mass and adsorbent mass must be positive")
    return 100 * np.asarray(q_grid, float) * masses[None, :] / solute_mass[:, None]


def sign(values):
    values = np.asarray(values, float)
    # Numerical closure at zero is not a positive or negative contribution.
    tolerance = 1e-11 * max(1., float(np.max(np.abs(values))))
    return np.where(np.abs(values) <= tolerance, 0, np.sign(values)).astype(int)


def log_outputs(q, solute_mass, masses):
    q, solute_mass, masses = map(np.asarray, (q, solute_mass, masses))
    if (q.shape != (len(solute_mass), len(masses))
            or not all(np.isfinite(a).all() for a in (q, solute_mass, masses))
            or any(np.any(a <= 0) for a in (q, solute_mass, masses))):
        raise ValueError("Positive finite uptake, initial solute mass and solid mass required")
    normalizer = np.log(solute_mass)[:, None] - np.log(masses)[None, :]
    uptake = np.log(q)
    return uptake, uptake - normalizer, normalizer


def analytic_mass_normalizer(masses, joint_weights, mass_ids):
    masses, joint_weights = np.asarray(masses, float), np.asarray(joint_weights, float)
    if (masses.ndim != 1 or np.any(masses <= 0) or not np.isfinite(masses).all()
            or joint_weights.ndim != 2 or joint_weights.shape[1] != len(masses)
            or np.any(joint_weights < 0) or not np.isclose(joint_weights.sum(), 1)):
        raise ValueError("Positive masses and normalized empirical weights required")
    log_mass = np.log(masses)
    return -(log_mass[np.asarray(mass_ids)] - joint_weights.sum(axis=0) @ log_mass)


def analytic_coalitions(dose, concentration, background_dose, background_concentration):
    dose, concentration = np.asarray(dose, float), np.asarray(concentration, float)
    bd, bc = np.asarray(background_dose, float), np.asarray(background_concentration, float)
    if np.any(dose <= 0) or np.any(concentration <= 0) or np.any(bd <= 0) or np.any(bc <= 0):
        raise ValueError("Positive dose and concentration are required")
    if dose.shape != concentration.shape or bd.shape != bc.shape:
        raise ValueError("Dose and concentration arrays must match")
    mean_c, mean_inverse_dose = bc.mean(), (1 / bd).mean()
    ceiling = np.column_stack((np.full(len(dose), np.mean(bc / bd)), mean_c / dose,
                               concentration * mean_inverse_dose, concentration / dose))
    inverse = np.column_stack((np.full(len(dose), mean_c * mean_inverse_dose), mean_c / dose,
                               np.full(len(dose), mean_c * mean_inverse_dose), mean_c / dose))
    return {"ceiling": ceiling, "inverse_dose": inverse}


def contributions(coalitions):
    empty, dose, context, full = coalitions.T
    phi_dose = ((dose - empty) + (full - context)) / 2
    phi_context = ((context - empty) + (full - dose)) / 2
    np.testing.assert_allclose(phi_dose + phi_context, full - empty, rtol=1e-12, atol=1e-10)
    return phi_dose, phi_context


def direction(value):
    return 1 if value > 1e-10 else -1 if value < -1e-10 else 0


def all_values_support(low, high, expected):
    lower = high.min() - low.max()
    upper = high.max() - low.min()
    if expected > 0:
        return lower > 1e-10
    if expected < 0:
        return upper < -1e-10
    return abs(lower) <= 1e-10 and abs(upper) <= 1e-10


def observed_pairs(data):
    fields = [column for column in data.columns if column not in ("ID", DOSE, UPTAKE)]
    pairs, members = [], []
    series_id = 0
    for _, group in data.groupby(fields, dropna=False, sort=False):
        doses = sorted(group[DOSE].unique())
        if len(doses) < 2:
            continue
        series_id += 1
        for index, row in group.iterrows():
            members.append(dict(series_id=series_id, source_index=int(index),
                                source_excel_row=int(index + 2), source_id=int(row.ID),
                                dose_g_l=float(row[DOSE]), c0_mg_l=float(row[C0]),
                                uptake_mg_g=float(row[UPTAKE]),
                                derived_removal_percent=float(100 * row[DOSE] * row[UPTAKE] / row[C0])))
        for low_dose, high_dose in zip(doses[:-1], doses[1:]):
            low = group.loc[group[DOSE] == low_dose]
            high = group.loc[group[DOSE] == high_dose]
            q_low, q_high = low[UPTAKE].to_numpy(float), high[UPTAKE].to_numpy(float)
            concentration = float(group[C0].iloc[0])
            r_low, r_high = 100 * low_dose * q_low / concentration, 100 * high_dose * q_high / concentration
            dq = float(q_high.mean() - q_low.mean())
            dr = float(r_high.mean() - r_low.mean())
            pair = dict(pair_id=len(pairs) + 1, series_id=series_id,
                        reported_reference_label=group.Reference.iloc[0],
                        reported_material_label=group["Adsorbent code"].iloc[0],
                        source_year=int(group.Year.iloc[0]), pollutant=group["Heavy metal"].iloc[0],
                        c0_mg_l=concentration, low_dose_g_l=float(low_dose), high_dose_g_l=float(high_dose),
                        low_source_indices=low.index.tolist(), high_source_indices=high.index.tolist(),
                        low_response_rows=len(low), high_response_rows=len(high),
                        low_observed_q_mg_g=float(q_low.mean()), high_observed_q_mg_g=float(q_high.mean()),
                        low_observed_q_min=float(q_low.min()), low_observed_q_max=float(q_low.max()),
                        high_observed_q_min=float(q_high.min()), high_observed_q_max=float(q_high.max()),
                        low_observed_removal_percent=float(r_low.mean()), high_observed_removal_percent=float(r_high.mean()),
                        observed_q_delta_mg_g=dq, observed_removal_delta_percentage_points=dr,
                        observed_q_direction=direction(dq), observed_removal_direction=direction(dr),
                        observed_q_relative_change=dq / float(q_low.mean()) if q_low.mean() != 0 else None,
                        q_direction_supported_by_all_recorded_values=bool(all_values_support(q_low, q_high, direction(dq))),
                        removal_direction_supported_by_all_recorded_values=bool(all_values_support(r_low, r_high, direction(dr))),
                        nominal_physical_observations=bool(np.all(np.r_[r_low, r_high] >= 0)
                                                          and np.all(np.r_[r_low, r_high] <= 100 + 1e-6)))
            pairs.append(pair)
    return pairs, members, fields


def balance_fields(c0, cf, mass, volume, observed_q, predicted_q):
    c0, cf, mass, volume, observed_q, predicted_q = [np.asarray(x, float)
        for x in (c0, cf, mass, volume, observed_q, predicted_q)]
    if len({x.shape for x in (c0, cf, mass, volume, observed_q, predicted_q)}) != 1:
        raise ValueError("All record arrays must have the same shape")
    if np.any(c0 <= 0) or np.any(mass <= 0) or np.any(volume <= 0):
        raise ValueError("Initial concentration, mass and volume must be positive")
    dose = mass / volume
    predicted_ct = c0 - dose * predicted_q
    physical_observation = np.isfinite(cf) & (cf >= 0) & (cf <= c0)
    # The analytic Ct=0 reference can leave a machine-roundoff subtraction residual.
    roundoff = 32 * np.finfo(float).eps * c0
    physical_prediction = (np.isfinite(predicted_ct) & (predicted_ct >= -roundoff)
                           & (predicted_ct <= c0 + roundoff))
    positive_ct = physical_observation & (cf > 0)
    identity_defined = positive_ct & (observed_q > 0)
    ct_relative = np.full(c0.shape, np.nan)
    amplification = np.full(c0.shape, np.nan)
    q_relative = np.full(c0.shape, np.nan)
    np.divide(abs(predicted_ct - cf), cf, out=ct_relative, where=positive_ct)
    np.divide(c0 - cf, cf, out=amplification, where=positive_ct)
    np.divide(abs(predicted_q - observed_q), observed_q, out=q_relative, where=observed_q > 0)
    return pd.DataFrame(dict(dose_g_l=dose, observed_c0_mg_l=c0,
                             observed_ct_mg_l=cf, observed_q_mg_g=observed_q,
                             predicted_q_mg_g=predicted_q, predicted_ct_mg_l=predicted_ct,
                             balance_audit_residual_mg_l=c0 - dose * observed_q - cf,
                             physical_observation=physical_observation,
                             physical_prediction=physical_prediction,
                             prediction_balance_roundoff_tolerance_mg_l=roundoff,
                             observed_removal_fraction=1 - cf / c0,
                             ct_absolute_error_mg_l=abs(predicted_ct - cf),
                             ct_absolute_error_over_c0=abs(predicted_ct - cf) / c0,
                             q_absolute_relative_error=q_relative,
                             ct_absolute_relative_error=ct_relative,
                             ct_relative_error_defined=positive_ct,
                             error_amplification=amplification,
                             error_identity_defined=identity_defined))


def pooled_r2(true, predicted):
    denominator = float(np.square(true - np.mean(true)).sum())
    return float(1 - np.square(true - predicted).sum() / denominator) if denominator > 0 else np.nan


def summary(frame, method, population):
    if len(frame) == 0:
        return dict(method=method, population=population, records=0)
    positive = frame[frame.ct_relative_error_defined]
    return dict(method=method, population=population, records=len(frame),
                q_r2=pooled_r2(frame.observed_q_mg_g.to_numpy(), frame.predicted_q_mg_g.to_numpy()),
                log_q_r2=pooled_r2(np.log(frame.observed_q_mg_g.to_numpy()), frame.predicted_log_q.to_numpy()),
                q_mae_mg_g=float(abs(frame.predicted_q_mg_g - frame.observed_q_mg_g).mean()),
                ct_mae_mg_l=float(frame.ct_absolute_error_mg_l.mean()),
                ct_rmse_mg_l=float(np.sqrt(np.square(frame.predicted_ct_mg_l - frame.observed_ct_mg_l).mean())),
                ct_r2=pooled_r2(frame.observed_ct_mg_l.to_numpy(), frame.predicted_ct_mg_l.to_numpy()),
                ct_mae_over_c0_percent=float(100 * frame.ct_absolute_error_over_c0.mean()),
                nonphysical_ct_predictions=int((~frame.physical_prediction).sum()),
                nonphysical_ct_prediction_percent=float(100 * (~frame.physical_prediction).mean()),
                positive_observed_ct_records=len(positive),
                zero_observed_ct_records=int(frame.observed_ct_mg_l.eq(0).sum()),
                negative_observed_ct_records=int(frame.observed_ct_mg_l.lt(0).sum()),
                median_q_relative_error_percent=float(100 * frame.q_absolute_relative_error.median()),
                median_ct_relative_error_percent=float(100 * positive.ct_absolute_relative_error.median()),
                p90_ct_relative_error_percent=float(100 * positive.ct_absolute_relative_error.quantile(.9)),
                median_error_amplification=float(positive.error_amplification.median()))


def scan_response(predict_q, dose, bounds):
    if (not np.isfinite([dose, *bounds]).all() or dose <= 0
            or bounds[0] <= 0 or bounds[1] <= bounds[0]):
        raise ValueError("Expected positive dose and increasing positive C0 bounds")
    c0 = np.geomspace(*bounds, SCAN_POINTS)
    q = np.asarray(predict_q(c0), float)
    if q.shape != c0.shape:
        raise ValueError("Expected one uptake prediction per initial concentration")
    ct = c0 - dose * q
    finite = np.isfinite(q) & np.isfinite(ct)
    physical = finite & (q >= 0) & (ct >= 0) & (ct <= c0)
    return pd.DataFrame(dict(c0_mg_l=c0, predicted_q_mg_g=q,
                             predicted_ct_mg_l=ct, finite=finite, physical=physical))


def solve_target(predict_q, target, dose, scan):
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Expected positive finite residual concentration")
    c0 = scan.c0_mg_l.to_numpy()
    q = scan.predicted_q_mg_g.to_numpy()
    ct = scan.predicted_ct_mg_l.to_numpy()
    physical = scan.physical.to_numpy(bool)
    delta = ct - target
    result = dict(status="", c0_mg_l=np.nan, predicted_q_mg_g=np.nan,
                  balance_residual_mg_l=np.nan, roots_json="[]", roots_detected=0,
                  nonfinite_scan_points=int((~scan.finite).sum()),
                  nonphysical_finite_scan_points=int((scan.finite & ~scan.physical).sum()),
                  strictly_increasing_finite_scan=bool(scan.finite.all()
                                                     and np.all(np.diff(ct) > 0)),
                  root_uniqueness_proven=False)
    roots = c0[physical & (np.abs(delta) <= 1e-10)].tolist()
    brackets = physical[:-1] & physical[1:] & (delta[:-1] * delta[1:] < 0)

    def residual(value):
        uptake = np.asarray(predict_q(np.array([value])), float)
        if uptake.shape != (1,) or not np.isfinite(uptake[0]):
            raise ValueError("A root query returned an invalid uptake prediction")
        return value - dose * uptake[0] - target

    failed_brackets = 0
    for index in np.flatnonzero(brackets):
        try:
            roots.append(float(brentq(residual, c0[index], c0[index + 1],
                                      xtol=1e-10, rtol=1e-12)))
        except ValueError:
            failed_brackets += 1
    unique = []
    for root in sorted(roots):
        if not unique or not np.isclose(root, unique[-1], atol=1e-8, rtol=1e-10):
            unique.append(float(root))
    checked = []
    for root in unique:
        uptake = float(np.asarray(predict_q(np.array([root])), float)[0])
        residual_ct = root - dose * uptake
        if (np.isfinite([uptake, residual_ct]).all() and uptake > 0
                and 0 <= residual_ct <= root and c0[0] <= root <= c0[-1]
                and abs(residual_ct - target) < 1e-7):
            checked.append(dict(c0_mg_l=root, predicted_q_mg_g=uptake,
                                balance_residual_mg_l=float(residual_ct - target)))
    result.update(roots_json=json.dumps(checked), roots_detected=len(checked),
                  failed_brackets=failed_brackets)
    if failed_brackets or len(checked) != len(unique):
        result["status"] = "invalid_root_evaluation"
    elif len(checked) > 1:
        result["status"] = "multiple_physical_roots_detected"
    elif len(checked) == 1:
        result.update(status="one_physical_root_detected", **checked[0])
    elif not scan.finite.any():
        result["status"] = "nonfinite_prediction"
    elif not physical.any():
        result["status"] = "no_physical_scan_values"
    else:
        result["status"] = "no_physical_root_detected_in_bounds"
    return result


def capacity(dose, alpha):
    dose = np.asarray(dose, dtype=float)
    if (not np.isfinite(dose).all() or np.any(dose <= 0)
            or not np.isfinite(alpha) or np.any(1 + alpha * dose <= 0)):
        raise ValueError("Expected a positive finite reciprocal capacity on the supplied dose range")
    return QMAX / (1 + alpha * dose)


def relation(ce, dose, alpha):
    ce, dose = np.broadcast_arrays(np.asarray(ce, dtype=float), np.asarray(dose, dtype=float))
    if not np.isfinite(ce).all() or np.any(ce < 0):
        raise ValueError("Expected finite nonnegative equilibrium concentration")
    return capacity(dose, alpha) * K * ce / (1 + K * ce)


def solve(x, alpha):
    x = np.asarray(x, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 2 or not len(x)
            or not np.isfinite(x).all() or np.any(x <= 0)):
        raise ValueError("Expected positive finite C0 and dose pairs")
    c0, dose = x.T
    cap = capacity(dose, alpha)
    b = 1 + K * dose * cap - K * c0
    discriminant = np.hypot(b, 2 * np.sqrt(K * c0))
    ce = np.empty(len(x))
    positive = b >= 0
    ce[positive] = 2 * c0[positive] / (b[positive] + discriminant[positive])
    ce[~positive] = (discriminant[~positive] - b[~positive]) / (2 * K)
    q = cap * K * ce / (1 + K * ce)
    h_ce = cap * K / (1 + K * ce) ** 2
    h_dose = -alpha * q / (1 + alpha * dose)
    path_dose = (h_dose - q * h_ce) / (1 + dose * h_ce)
    eta = dose * h_dose / q
    coupling = dose * h_ce
    return dict(
        ce=ce, q=q, removal=100 * dose * q / c0, h_ce=h_ce, h_dose=h_dose,
        q_dose_at_fixed_c0=path_dose, common_state_dose_elasticity=eta,
        depletion_coupling=coupling, operating_dose_elasticity=dose * path_dose / q,
        balance_error=ce + dose * q - c0,
    )


