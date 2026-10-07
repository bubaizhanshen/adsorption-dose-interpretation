"""Recompute response-attribution and recorded-dose sensitivity analyses."""

from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'reproducibility' if HERE.name == 'source' else HERE.parent
OUT = PACKAGE / 'results/review_audit'
DATA = PACKAGE / 'data'
REFERENCE = None
CACHE = PACKAGE / '.cache/review_audit'
os.umask(0o077)
CACHE.mkdir(parents=True, mode=0o700, exist_ok=True)
for name in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'MPLCONFIGDIR'):
    os.environ[name] = str(CACHE)
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(PACKAGE / 'src'))

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import physics as p
from models import numpy_prediction


def read(name, data=False):
    root = DATA if data else REFERENCE
    return None if root is None else pd.read_csv(root / name, float_precision='round_trip')


def save(frame, name):
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def load_biochar():
    cfg = json.loads((DATA / 'model_settings.json').read_text())['biochar']
    frame = read('biochar_records.csv', True)
    order = np.random.RandomState(cfg['split_seed']).permutation(len(frame))
    train, test = order[:cfg['training_records']], order[cfg['training_records']:]
    with np.load(DATA / 'biochar_weights.npz', allow_pickle=False) as f:
        weights = {k: f[k] for k in f.files}
    return cfg, frame, train, test, weights


def bio_predict(frame, cfg, weights):
    return np.exp(np.concatenate([numpy_prediction(frame.iloc[i:i+128],
                 cfg['numeric_features'], cfg['categorical_features'], cfg['vocabulary'], weights)
                 for i in range(0, len(frame), 128)]))


def load_case(name):
    if name == 'biochar':
        cfg, frame, train, test, weights = load_biochar()
        features = cfg['numeric_features'] + cfg['categorical_features']
        variable = 'loading (g)'
        predict = lambda x: bio_predict(x, cfg, weights)
        pollutant, balance = 'inorganics', 'Adsorbent'
        split = np.full(len(frame), 'test', dtype=object)
        split[train] = 'train'
    else:
        cfg = json.loads((DATA / 'model_settings.json').read_text())[name]
        frame = read('activated_carbon_records.csv', True)
        features, variable = cfg['features'], 'Dose (g/L)'
        bins = pd.qcut(frame[p.UPTAKE], q=29, labels=False, duplicates='drop')
        train, temp = train_test_split(np.arange(len(frame)), test_size=.3, random_state=42, stratify=bins)
        validation, test = train_test_split(temp, test_size=.5, random_state=42, stratify=bins[temp])
        model = make_pipeline(StandardScaler(), GradientBoostingRegressor(**cfg['parameters']))
        model.fit(frame.iloc[train][features].to_numpy(float), frame.iloc[train][p.UPTAKE])
        predict = lambda x: model.predict(x[features].to_numpy(float))
        pollutant, balance = 'Heavy metal', 'Reference'
        split = np.full(len(frame), 'train', dtype=object)
        split[validation], split[test] = 'validation', 'test'
    fields = [f for f in features if f != variable]
    tuples = pd.Series([tuple(r) for r in frame[fields].to_numpy()], dtype=object)
    cids, _ = pd.factorize(tuples, sort=False)
    contexts = frame[fields].drop_duplicates().reset_index(drop=True)
    values, vids = np.unique(frame[variable].to_numpy(float), return_inverse=True)
    inputs = contexts.loc[contexts.index.repeat(len(values))].reset_index(drop=True)
    inputs[variable] = np.tile(values, len(contexts))
    grid = predict(inputs).reshape(len(contexts), len(values))
    numerator = (contexts.Ci * contexts['Volume (L)']).to_numpy() if name == 'biochar' else contexts[p.C0].to_numpy()
    ceiling = numerator[:, None] / values[None, :]
    original = read('author_output_game_records.csv' if name == 'biochar' else 'activated_carbon_output_game_records.csv')
    if original is not None:
        np.testing.assert_allclose(grid[cids[test], vids[test]], original.predicted_q_mg_g, rtol=1e-10, atol=1e-8)
    return dict(name=name, frame=frame, cfg=cfg, train=train, test=test, features=features,
                variable=variable, pollutant=pollutant, balance=balance, split=split,
                contexts=contexts, cids=cids, vids=vids, values=values, grid=grid, ceiling=ceiling)


def joint(case, rows, weights):
    matrix = np.zeros_like(case['grid'])
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    np.add.at(matrix, (case['cids'][rows], case['vids'][rows]), weights)
    return matrix


def game(case, grid, background, query):
    return p.grouped_contributions(grid, background, case['cids'][query], case['vids'][query])


def sign(values):
    return np.where(np.abs(values) <= 1e-8, 0, np.sign(values)).astype(int)


def audit_case(case):
    frame, train, test = case['frame'], case['train'], case['test']
    q, ceiling = case['grid'], case['ceiling']
    removal = 100 * q / ceiling
    invalid = (removal < -1e-8) | (removal > 100 + 1e-8)
    original_bg = joint(case, train, np.ones(len(train)))
    backgrounds = [('training_rows', train, np.ones(len(train)), test)]
    profiles = frame.iloc[train][case['features']].astype(str).agg('|'.join, axis=1)
    backgrounds.append(('unique_input_profiles', train, 1 / profiles.map(profiles.value_counts()).to_numpy(), test))
    labels = frame.iloc[train][case['balance']]
    backgrounds.append(('material_label_balanced' if case['name'] == 'biochar' else 'source_label_balanced',
                        train, 1 / labels.map(labels.value_counts()).to_numpy(), test))
    for pol in frame.iloc[test][case['pollutant']].unique():
        bgrows = train[frame.iloc[train][case['pollutant']].eq(pol).to_numpy()]
        queries = test[frame.iloc[test][case['pollutant']].eq(pol).to_numpy()]
        assert len(bgrows)
        backgrounds.append(('same_pollutant', bgrows, np.ones(len(bgrows)), queries))
    records = []
    for name, rows, weight, query in backgrounds:
        background = joint(case, rows, weight)
        outputs = {label: game(case, grid, background, query)
                   for label, grid in [('uptake', q), ('removal', removal), ('ceiling', ceiling)]}
        for i, row in enumerate(query):
            records.append(dict(case=case['name'], background=name, source_excel_row=int(row+2),
                                variable=float(frame.iloc[row][case['variable']]),
                                pollutant=frame.iloc[row][case['pollutant']], background_rows=len(rows),
                                uptake_phi=float(outputs['uptake']['mass_phi'][i]),
                                removal_phi=float(outputs['removal']['mass_phi'][i]),
                                ceiling_phi=float(outputs['ceiling']['mass_phi'][i]),
                                uptake_baseline=outputs['uptake']['baseline'],
                                uptake_variable_coalition=float(outputs['uptake']['mass_coalition'][i]),
                                uptake_other_coalition=float(outputs['uptake']['context_coalition'][i]),
                                removal_baseline=outputs['removal']['baseline'],
                                removal_variable_coalition=float(outputs['removal']['mass_coalition'][i]),
                                removal_other_coalition=float(outputs['removal']['context_coalition'][i]),
                                inverse_threshold=float(1 / np.sum(background.sum(axis=0) / case['values']))))
    records = pd.DataFrame(records)
    records['sign_reversed'] = sign(records.uptake_phi) != sign(records.removal_phi)
    records['ceiling_sign_agrees'] = sign(records.uptake_phi) == sign(records.ceiling_phi)
    save(records, case['name'] + '_background_records.csv')
    summary = records.groupby('background', sort=False).agg(records=('source_excel_row', 'size'),
        sign_reversals=('sign_reversed', 'sum'), ceiling_sign_agreement=('ceiling_sign_agrees', 'sum')).reset_index()
    summary.insert(0, 'case', case['name'])
    save(summary, case['name'] + '_background_summary.csv')
    # Each coalition retains the original training frequencies and equal query weights.
    query_bg = joint(case, test, np.ones(len(test)))
    distributions = dict(empty=original_bg, variable=np.outer(original_bg.sum(axis=1), query_bg.sum(axis=0)),
                         other=np.outer(query_bg.sum(axis=1), original_bg.sum(axis=0)), full=query_bg)
    coalition_rows = []
    violation = np.maximum(-removal, 0) + np.maximum(removal-100, 0)
    for label, weights in distributions.items():
        keep = weights > 0
        values, w = violation[keep], weights[keep]
        order = np.argsort(values)
        quantiles = np.interp([.5, .95, .99], np.cumsum(w[order]), values[order])
        coalition_rows.append(dict(case=case['name'], coalition=label,
            weighted_nonphysical_fraction=float(np.sum(weights * invalid)),
            mean_exceedance_percentage_points=float(np.sum(weights * violation)),
            median_exceedance=quantiles[0], p95_exceedance=quantiles[1], p99_exceedance=quantiles[2],
            maximum_exceedance=float(values.max()), unique_weighted_grid_cells=int(keep.sum()),
            total_grid_cells=int(q.size), total_nonphysical_grid_cells=int(invalid.sum())))
    save(pd.DataFrame(coalition_rows), case['name'] + '_hybrid_coalitions.csv')
    transformations = {'original': q, 'bounded_function': np.clip(q, 0, ceiling)}
    sensitivities, attributed_invalid, quadrants = [], [], []
    for label, grid in transformations.items():
        gq = game(case, grid, original_bg, test)
        gr = game(case, 100 * grid / ceiling, original_bg, test)
        for i, row in enumerate(test):
            sensitivities.append(dict(case=case['name'], function=label, source_excel_row=int(row+2),
                uptake_phi=gq['mass_phi'][i], removal_phi=gr['mass_phi'][i],
                predicted_uptake=gq['predicted'][i], predicted_removal=gr['predicted'][i],
                sign_reversed=bool(sign(gq['mass_phi'])[i] != sign(gr['mass_phi'])[i])))
        if label == 'original':
            for quantity, full in [('uptake', q), ('removal', removal)]:
                a = game(case, full, original_bg, test)
                b = game(case, np.where(invalid, full, 0), original_bg, test)
                c = game(case, np.where(invalid, 0, full), original_bg, test)
                np.testing.assert_allclose(a['mass_phi'], b['mass_phi'] + c['mass_phi'], atol=1e-9)
                for i, row in enumerate(test):
                    attributed_invalid.append(dict(case=case['name'], response=quantity, source_excel_row=int(row+2),
                        full_contribution=a['mass_phi'][i], contribution_through_nonphysical_values=b['mass_phi'][i],
                        contribution_through_physical_values=c['mass_phi'][i]))
            for qs in (-1, 0, 1):
                for rs in (-1, 0, 1):
                    quadrants.append(dict(case=case['name'], scale='raw', uptake_sign=qs, removal_sign=rs,
                        records=int(((sign(gq['mass_phi']) == qs) & (sign(gr['mass_phi']) == rs)).sum())))
    if case['name'] == 'biochar':
        gq = game(case, np.log(q), original_bg, test)
        gr = game(case, np.log(removal/100), original_bg, test)
        for qs in (-1, 0, 1):
            for rs in (-1, 0, 1):
                quadrants.append(dict(case=case['name'], scale='log', uptake_sign=qs, removal_sign=rs,
                    records=int(((sign(gq['mass_phi']) == qs) & (sign(gr['mass_phi']) == rs)).sum())))
    save(pd.DataFrame(sensitivities), case['name'] + '_bounded_function_records.csv')
    save(pd.DataFrame(attributed_invalid), case['name'] + '_nonphysical_contributions.csv')
    save(pd.DataFrame(quadrants), case['name'] + '_response_quadrants.csv')
    np.savez_compressed(OUT / (case['name'] + '_grid.npz'), q=q, ceiling=ceiling,
                        context_ids=case['cids'], variable_ids=case['vids'], values=case['values'],
                        training_joint_weights=original_bg, train=train, test=test)
    registry = frame.copy()
    registry.insert(0, 'source_excel_row', np.arange(len(frame)) + 2)
    registry.insert(1, 'split', case['split'])
    save(registry, case['name'] + '_record_registry.csv')
    return summary.to_dict('records')


def controls():
    background = np.array([(c, d) for c in (50., 100., 200.) for d in np.geomspace(.1, 4., 201)])
    rows = []
    for alpha in (-.05, -.02, 0., .02, .05, .1):
        base = p.solve(background, alpha)['q'].mean()
        for dose in (1., 2.):
            query = np.array([[100., dose]])
            output = p.solve(query, alpha)['q'][0]
            v_dose = p.solve(np.column_stack((background[:, 0], np.full(len(background), dose))), alpha)['q'].mean()
            v_c0 = p.solve(np.column_stack((np.full(len(background), 100.), background[:, 1])), alpha)['q'].mean()
            phi = .5 * (v_dose-base+output-v_c0)
            common = float(p.relation(30., dose, alpha))
            ce = brentq(lambda c: c + dose * float(p.relation(c, dose, alpha))-100., 0, 100., xtol=1e-12)
            np.testing.assert_allclose(output, p.relation(ce, dose, alpha), atol=1e-10)
            rows.append(dict(alpha_l_g=alpha, dose_g_l=dose, c0_mg_l=100., common_ce_mg_l=30.,
                             operating_ce_mg_l=ce, fixed_c0_uptake=output, common_ce_uptake=common,
                             dose_phi=phi, baseline=base, dose_coalition=v_dose, c0_coalition=v_c0))
    result = pd.DataFrame(rows)
    save(result, 'control_attributions.csv')
    high = result[result.dose_g_l.eq(2)].set_index('alpha_l_g')
    np.testing.assert_allclose(high.loc[[-.05, 0., .1], 'dose_phi'], [-23.131, -23.003, -22.982], atol=.001)
    assert (high.dose_phi < 0).all()
    return result.to_dict('records')


def pair_checks(case):
    frame = case['frame']
    raw, members, fields = p.observed_pairs(frame)
    pairs = pd.DataFrame(raw)
    original = read('activated_carbon_dose_pairs_records.csv')
    if original is not None:
        np.testing.assert_allclose(pairs.observed_q_delta_mg_g, original.observed_q_delta_mg_g)
        np.testing.assert_allclose(pairs.observed_removal_delta_percentage_points, original.observed_removal_delta_percentage_points)
    pairs['dose_ratio'] = pairs.high_dose_g_l / pairs.low_dose_g_l
    pairs['q_ratio'] = pairs.high_observed_q_mg_g / pairs.low_observed_q_mg_g
    np.testing.assert_allclose(pairs.high_observed_removal_percent / pairs.low_observed_removal_percent,
                               pairs.dose_ratio * pairs.q_ratio, rtol=1e-12)
    pairs['split_low'] = [json.dumps(case['split'][x].tolist()) for x in pairs.low_source_indices]
    pairs['split_high'] = [json.dumps(case['split'][x].tolist()) for x in pairs.high_source_indices]
    membership = pd.DataFrame(members).merge(frame.reset_index(names='source_index'), on='source_index', validate='many_to_one')
    membership['split'] = case['split'][membership.source_index]
    save(membership, 'dose_series_membership.csv')
    for key in ('low_source_indices', 'high_source_indices'):
        pairs[key.replace('indices', 'excel_rows')] = pairs[key].map(lambda v: json.dumps([int(x+2) for x in v]))
        pairs[key] = pairs[key].map(json.dumps)
    save(pairs, 'dose_pairs_complete.csv')
    directions = [(-1, 1), (-1, -1), (1, 1)]
    sensitivities = []
    for tol in (1e-10, 1e-6, 1e-3):
        dq = np.where(pairs.observed_q_delta_mg_g.abs() <= tol, 0, np.sign(pairs.observed_q_delta_mg_g))
        dr = np.where(pairs.observed_removal_delta_percentage_points.abs() <= tol, 0, np.sign(pairs.observed_removal_delta_percentage_points))
        for qs, rs in directions:
            sensitivities.append(dict(rule='absolute_direction_tolerance', value=tol, q_sign=qs, r_sign=rs,
                                      pair_count=int(((dq == qs) & (dr == rs)).sum())))
    save(pd.DataFrame(sensitivities), 'pair_tolerance_sensitivity.csv')
    balanced = []
    for qs, rs in directions:
        membership_flag = (pairs.observed_q_direction.eq(qs) & pairs.observed_removal_direction.eq(rs)).astype(float)
        for weighting, groups in [('pair', None), ('series', pairs.series_id), ('source_label', pairs.reported_reference_label)]:
            rate = membership_flag.mean() if groups is None else membership_flag.groupby(groups).mean().mean()
            balanced.append(dict(weighting=weighting, q_sign=qs, r_sign=rs, fraction=rate))
        for label in pairs.reported_reference_label.unique():
            mask = pairs.reported_reference_label.ne(label)
            balanced.append(dict(weighting='leave_one_source_label_out', omitted_source=label,
                                 q_sign=qs, r_sign=rs, fraction=membership_flag[mask].mean()))
    save(pd.DataFrame(balanced), 'pair_weighting_sensitivity.csv')
    duplicates = frame.duplicated(subset=[c for c in frame if c != 'ID'], keep=False)
    dup = frame[duplicates].copy()
    dup.insert(0, 'source_excel_row', dup.index + 2)
    dup.insert(1, 'split', case['split'][dup.index])
    save(dup, 'duplicate_records.csv')
    dedup = frame.drop_duplicates(subset=[c for c in frame if c != 'ID'])
    dp = pd.DataFrame(p.observed_pairs(dedup)[0])
    save(dp, 'deduplicated_dose_pairs.csv')
    result = dict(matching_fields=fields, missing_matching_cells=int(frame[fields].isna().sum().sum()),
                  pairs=len(pairs), series=pairs.series_id.nunique(), raw_rows=len(membership),
                  range_stable_q=int(pairs.q_direction_supported_by_all_recorded_values.sum()),
                  range_stable_removal=int(pairs.removal_direction_supported_by_all_recorded_values.sum()),
                  unique_duplicate_excess=int(frame.duplicated(subset=[c for c in frame if c != 'ID']).sum()),
                  duplicate_member_rows=int(duplicates.sum()), deduplicated_pairs=len(dp),
                  deduplicated_counts=[int((dp.observed_q_direction.eq(qs) & dp.observed_removal_direction.eq(rs)).sum()) for qs, rs in directions])
    return result


def high_removal(case):
    ids = case['test']
    frame = case['frame'].iloc[ids].reset_index(drop=True)
    observed = frame.Ci - frame['loading (g)'] * frame['qe'] / frame['Volume (L)']
    prediction = case['grid'][case['cids'][ids], case['vids'][ids]]
    records = p.balance_fields(frame.Ci, observed, frame['loading (g)'],
                               frame['Volume (L)'], frame['qe'], prediction)
    records['source_excel_row'] = ids + 2
    records['method'] = 'original_checkpoint'
    assert len(records) == 456 and records.source_excel_row.is_unique
    records['material_label'] = frame.Adsorbent
    records['pollutant'] = frame.inorganics
    selected = records[records.physical_observation & records.observed_removal_fraction.ge(.9)].copy()
    assert len(selected) == 44
    save(selected, 'high_removal_records.csv')
    save(selected.groupby(['material_label', 'pollutant'], sort=False).size().rename('records').reset_index(), 'high_removal_composition.csv')
    save(records[~records.physical_observation], 'nonphysical_biochar_observation.csv')
    return dict(high_removal_records=len(selected), materials=selected.material_label.nunique(),
                pollutants=selected.pollutant.nunique(), columns=records.columns.tolist())


def query_support():
    cfg, frame, train, test, weights = load_biochar()
    features = cfg['numeric_features'] + cfg['categorical_features']
    other = [f for f in features if f != 'Ci']
    split = np.full(len(frame), 'test', dtype=object)
    split[train] = 'train'
    support, states, endpoints = [], [], []
    for index in (38, 39):
        assert index in test
        row = frame.iloc[index]
        source = frame.Adsorbent.eq(row.Adsorbent) & frame.inorganics.eq(row.inorganics)
        exact = frame[other].eq(row[other]).all(axis=1)
        bounds = (float(frame.loc[source, 'Ci'].min()), float(frame.loc[source, 'Ci'].max()))
        assert bounds == (1., 10.)
        dose = float(row['loading (g)']/row['Volume (L)'])
        for label, mask in [('same_material_and_pollutant', source),
                            ('identical_nonconcentration_inputs', exact)]:
            for subset, subset_mask in [('all', np.ones(len(frame), bool)), ('train', split == 'train')]:
                selected = frame[mask & subset_mask]
                support.append(dict(source_excel_row=index+2, dose_g_l=dose, match_definition=label,
                    split=subset, matching_rows=len(selected), distinct_c0=selected.Ci.nunique(),
                    minimum_c0=selected.Ci.min(), maximum_c0=selected.Ci.max(),
                    member_excel_rows=json.dumps((selected.index+2).tolist())))
        def predict(c0):
            query = pd.DataFrame([row.to_dict()] * len(c0))
            query['Ci'] = np.asarray(c0)
            return bio_predict(query, cfg, weights)
        scan = p.scan_response(predict, dose, bounds)
        states.append(scan.assign(source_excel_row=index+2, dose_g_l=dose))
        for target in (1., 2.5, 5., 7.5):
            result = p.solve_target(predict, target, dose, scan)
            endpoints.append(dict(source_excel_row=index+2, dose_g_l=dose,
                                  target_ct_mg_l=target, **result))
    solved = pd.DataFrame(endpoints).sort_values(['target_ct_mg_l', 'dose_g_l']).reset_index(drop=True)
    old = read('author_query_endpoints.csv')
    if old is not None:
        old = old.sort_values(['target_ct_mg_l', 'dose_g_l']).reset_index(drop=True)
        np.testing.assert_allclose(solved[['c0_mg_l', 'predicted_q_mg_g']],
                                   old[['c0_mg_l', 'predicted_q_mg_g']], atol=1e-7, rtol=1e-9)
    save(pd.DataFrame(support), 'query_local_support.csv')
    save(pd.concat(states, ignore_index=True), 'query_scan.csv')
    save(solved, 'query_roots.csv')
    return dict(endpoints=len(solved), maximum_balance_residual=float(solved.balance_residual_mg_l.abs().max()),
                all_one_root_detected=bool(solved.roots_detected.eq(1).all()),
                global_uniqueness_proven=False, support=support)


def monte_carlo_one(query_order):
    cfg, frame, train, test, weights = load_biochar()
    saved = read('biochar_saved_shap.csv', True)['loading (g)'].to_numpy()
    order = np.argsort(np.abs(saved), kind='stable')[:12]
    idx = int(order[query_order])
    query = frame.iloc[test[idx]]
    features = cfg['numeric_features'] + cfg['categorical_features']
    other = [f for f in features if f != 'loading (g)']
    results = []
    # A uniform position for the focal input samples the Shapley permutation distribution.
    for replicate in range(4):
        rng = np.random.default_rng(1007000 + 100 * query_order + replicate)
        n = 8192
        background_rows = rng.choice(train, size=n, replace=True)
        predecessor_probability = rng.random((n, 1))
        mask = rng.random((n, len(other))) < predecessor_probability
        absent = frame.iloc[background_rows][features].reset_index(drop=True).copy()
        for i, column in enumerate(other):
            absent.loc[mask[:, i], column] = query[column]
        present = absent.copy()
        present['loading (g)'] = query['loading (g)']
        differences = bio_predict(present, cfg, weights) - bio_predict(absent, cfg, weights)
        assert np.isfinite(differences).all()
        for size in (512, 2048, 8192):
            sample = differences[:size]
            results.append(dict(query_order=query_order, test_order=idx, source_excel_row=int(test[idx]+2),
                                loading_g=float(query['loading (g)']), saved_phi=float(saved[idx]),
                                replicate=replicate, permutation_draws=size, estimate=float(sample.mean()),
                                monte_carlo_se=float(sample.std(ddof=1)/np.sqrt(size))))
    result = pd.DataFrame(results)
    save(result, f'near_zero_sampling_{query_order:02d}.csv')
    return dict(query_order=query_order, source_excel_row=int(test[idx]+2), saved_phi=float(saved[idx]),
                estimate=float(result[result.permutation_draws.eq(8192)].estimate.mean()))


def worker_paths(data, output):
    global DATA, OUT
    DATA, OUT = Path(data), Path(output)


def main():
    global DATA, OUT, REFERENCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--skip-sampling', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--reference-dir', type=Path,
                        help='Optional reference results, used only for numerical comparisons')
    args = parser.parse_args()
    DATA, OUT, REFERENCE = args.data_dir, args.output, args.reference_dir
    if args.workers < 1:
        parser.error('--workers must be positive')
    if OUT.exists() and any(OUT.iterdir()):
        parser.error(f'Output directory must be empty: {OUT}')
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    bio, carbon = load_case('biochar'), load_case('activated_carbon')
    summary = {'backgrounds': audit_case(bio) + audit_case(carbon), 'controls': controls(),
               'pairs': pair_checks(carbon), 'high_removal': high_removal(bio),
               'query_support': query_support()}
    if not args.skip_sampling:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_paths,
                                 initargs=(DATA, OUT)) as pool:
            summary['near_zero_sampling'] = list(pool.map(monte_carlo_one, range(12)))
        combined = pd.concat([pd.read_csv(OUT/f'near_zero_sampling_{i:02d}.csv') for i in range(12)], ignore_index=True)
        save(combined, 'near_zero_sampling.csv')
    summary['near_zero_sampling_completed'] = not args.skip_sampling
    summary['saved_results_used_as_inputs'] = False
    summary['precision_basis'] = 'Fresh NumPy replay of unchanged biochar weights'
    summary['elapsed_seconds'] = time.monotonic() - started
    summary['sign_tolerance'] = 'absolute 1e-8 in each response unit; source counts checked separately'
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
