"""Recompute graphical bounds and supplied controls from portable inputs."""
import json
import os
from pathlib import Path
import sys

os.umask(0o077)
HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent if (HERE.parent / 'data/model_settings.json').exists() else HERE.parent / 'reproducibility'
sys.path.insert(0, str(PACKAGE / 'src'))
import numpy as np
import pandas as pd
import auxiliary_physics as a


def compare_columns(actual, expected, columns):
    assert len(actual) == len(expected)
    maximum = 0.0
    for column in columns:
        x, y = actual[column].to_numpy(float), expected[column].to_numpy(float)
        np.testing.assert_allclose(x, y, atol=1e-8, rtol=1e-9, equal_nan=True, err_msg=column)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            maximum = max(maximum, float(np.max(np.abs(x[finite] - y[finite]))))
    return maximum


def graphical(data, saved, output):
    calibration = json.loads((data / 'zhao_calibration.json').read_text())
    raw = pd.read_csv(data / 'zhao_marker_pixels.csv')
    assert not raw.duplicated(['metal', 'dose_g_l', 'marker']).any()
    rows = []
    radius = calibration['reading_radius_pixels']
    for row in raw.to_dict('records'):
        panel = calibration['panels'][row['metal']]
        def scale(values, axis):
            ticks, units = panel[axis + '_ticks'], panel[axis + '_values']
            return units[0] + (np.asarray(values) - ticks[0]) * (units[1] - units[0]) / (ticks[1] - ticks[0])
        x, y = row['x_pixel'], row['y_pixel']
        assert row['visible_top_pixel'] <= y <= row['visible_bottom_pixel']
        xl, xu = sorted(scale([x - radius, x + radius], 'x'))
        yl, yu = sorted(scale([y - radius, y + radius], 'y'))
        vl, vu = sorted(scale([row['visible_top_pixel'] - radius, row['visible_bottom_pixel'] + radius], 'y'))
        rows.append(dict(**row, ce_mg_l=float(scale(x, 'x')), q_mg_g=float(scale(y, 'y')),
                         ce_low=xl, ce_high=xu, q_low=yl, q_high=yu, visible_q_low=vl, visible_q_high=vu))
    markers = pd.DataFrame(rows)
    old_markers = pd.read_csv(saved / 'real_concave_profiles/zhao_full_markers.csv')
    marker_error = compare_columns(markers, old_markers, ['ce_mg_l', 'q_mg_g', 'ce_low', 'ce_high',
                                   'q_low', 'q_high', 'visible_q_low', 'visible_q_high'])
    markers.to_csv(output / 'zhao_markers.csv', index=False)
    families = {}
    for envelope in ('reading', 'visible_vertical_extent'):
        frame = markers.copy()
        if envelope != 'reading':
            frame['q_low'], frame['q_high'] = frame.visible_q_low, frame.visible_q_high
        for (metal, dose), curve in frame.groupby(['metal', 'dose_g_l']):
            families[envelope, metal, dose] = a.MonotoneRectangles(curve)
    queries = pd.read_csv(data / 'zhao_frozen_queries.csv')
    rows, checks = [], []
    for query in queries.to_dict('records'):
        if not np.isfinite(query['c0_mg_l']):
            rows.append(dict(**query, status='no_frozen_query'))
            continue
        low = families[query['envelope'], query['metal'], query['low_dose_g_l']]
        high = families[query['envelope'], query['metal'], query['high_dose_g_l']]
        args = query['c0_mg_l'], query['low_dose_g_l'], query['high_dose_g_l']
        bounds = a.component_bounds(low, high, *args)
        covered = (min(bounds['low_ce_lower'], bounds['high_ce_lower']) >= query['common_ce_lower'] - 1e-10
                   and max(bounds['low_ce_upper'], bounds['high_ce_upper']) <= query['common_ce_upper'] + 1e-10)
        status = ('operating_ce_outside_common_coverage' if not covered else
                  'mixed_c0_outside_original_range' if not bounds['mixed_states_within_original_c0'] else 'supported')
        rows.append(dict(**query, status=status, **bounds))
        for lw in low.witnesses():
            for hw in high.witnesses():
                result = a.linear_pair(lw, hw, *args)
                checks.append(dict(closure=result['closure_error'], balance=result['balance_error'],
                                   violation=a.bound_violation(bounds, result)))
    current = pd.DataFrame(rows)
    old = pd.read_csv(saved / 'real_concave_profiles/monotone_mediator_summary.csv')
    assert current.status.tolist() == old.status.tolist()
    columns = [c for c in current if c.endswith(('_lower', '_upper'))]
    bound_error = compare_columns(current, old, columns)
    witness = pd.DataFrame(checks)
    assert witness.to_numpy().max() < 1e-7
    current.to_csv(output / 'zhao_outer_bounds.csv', index=False)
    witness.to_csv(output / 'zhao_witness_checks.csv', index=False)
    return dict(markers=len(markers), queries=len(current), marker_error=marker_error,
                bound_error=bound_error, status_counts=current.status.value_counts().to_dict(),
                witness_pairs=len(witness), maximum_witness_errors=witness.max().to_dict())


def kinetic(saved, output):
    original = pd.read_csv(saved / 'real_concave_profiles/kinetic_shared_observations.csv')
    rows = []
    for case, group in original.groupby('case', sort=False):
        c0, dose = group.c0_mg_l.to_numpy(), group.dose_g_l.to_numpy()
        if case == 'equilibrium_limit':
            ct = c0 / (1 + a.HENRY_K * dose)
            q = a.HENRY_K * ct
        else:
            state = a.kinetic_state(c0, dose, float(case.removeprefix('kt_')) / a.LDF_K)
            ct, q = state['ct'], state['q']
        rows.extend(dict(case=case, c0_mg_l=c, dose_g_l=d, ct_mg_l=t, q_mg_g=u)
                    for c, d, t, u in zip(c0, dose, ct, q))
    observations = pd.DataFrame(rows)
    error = compare_columns(observations, original, ['c0_mg_l', 'dose_g_l', 'ct_mg_l', 'q_mg_g'])
    pairs, ode_error = [], 0.0
    for kt in (.01, .1, 1., 10., 100.):
        time = kt / a.LDF_K
        dose = np.array([1., 2.])
        state = a.kinetic_state(100., dose, time)
        common_q = 10 * a.apparent_slope(dose, time)
        required = 10 + dose * common_q
        np.testing.assert_allclose(a.kinetic_state(required, dose, time)['ct'], 10., atol=1e-10)
        pairs.append(dict(time_h=time, operating_q_change_percent=100*(state['q'][1]/state['q'][0]-1),
                          common_ct_q_change_percent=100*(common_q[1]/common_q[0]-1),
                          required_low_c0_mg_l=required[0], required_high_c0_mg_l=required[1]))
        for i, d in enumerate(dose):
            numerical = a.independent_ode(100., d, time)
            ode_error = max(ode_error, float(np.max(np.abs(numerical - [state['ct'][i], state['q'][i]]))))
    pairs = pd.DataFrame(pairs)
    pair_error = compare_columns(pairs, pd.read_csv(saved / 'real_concave_profiles/kinetic_shared_pairs.csv'), list(pairs))
    assert ode_error < 1e-7
    observations.to_csv(output / 'kinetic_observations.csv', index=False)
    pairs.to_csv(output / 'kinetic_pairs.csv', index=False)
    return dict(records=len(observations), pairs=len(pairs), observation_error=error,
                pair_error=pair_error, independent_ode_error=ode_error)


def competition(saved, output):
    rows, max_error = [], 0.0
    for b0 in (0., .5, 1., 2.):
        low, high = [a.equilibrate(1., b0, d) for d in (1., 2.)]
        for target in (.1, .5):
            states = [a.direct_at_shared_a(target, b0, d) for d in (1., 2.)]
            for state in states:
                prediction = lambda c0, dose: a.equilibrate(c0, b0, dose).qa_mmol_g
                root = a.query_at_shared_a(prediction, target, state.dose_g_l)
                assert root['status'] == 'supported'
                max_error = max(max_error, abs(root['a0_mmol_l']-state.a0_mmol_l))
                direct = a.equal_affinity_reference(state.a0_mmol_l, b0, state.dose_g_l)
                np.testing.assert_allclose(direct, [state.a_mmol_l, state.b_mmol_l], atol=1e-12)
            rows.append(dict(b0_mmol_l=b0, target_a_mmol_l=target,
                             operating_qa_change_percent=100*(high.qa_mmol_g/low.qa_mmol_g-1),
                             common_a_qa_change_percent=100*(states[1].qa_mmol_g/states[0].qa_mmol_g-1)))
    current = pd.DataFrame(rows)
    difference = compare_columns(current, pd.read_csv(saved / 'concave_equivalence/competitive_contrasts.csv'), list(current))
    current.to_csv(output / 'competitive_contrasts.csv', index=False)
    assert max_error < 1e-10
    return dict(comparisons=len(current), saved_difference=difference, independent_query_error=max_error)


def main():
    data, saved = PACKAGE / 'data', PACKAGE / 'results'
    output = saved / 'supplement_recomputed'
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    summary = dict(graphical=graphical(data, saved, output), kinetic=kinetic(saved, output),
                   competition=competition(saved, output))
    (output / 'checks.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
