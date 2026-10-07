"""Recompute primary comparisons from local numeric inputs and public-model weights."""

from pathlib import Path
import argparse
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]
os.umask(0o077)
CACHE = ROOT / '.cache'
CACHE.mkdir(mode=0o700, exist_ok=True)
for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'MPLCONFIGDIR'):
    os.environ[key] = str(CACHE)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from models import numpy_prediction
import physics as p


def read(name, inputs=False):
    return pd.read_csv(ROOT / ('data' if inputs else 'results') / name, float_precision='round_trip')


def predict(frame, cfg, weights):
    return np.exp(np.concatenate([
        numpy_prediction(frame.iloc[i:i+32], cfg['numeric_features'], cfg['categorical_features'],
                         cfg['vocabulary'], weights) for i in range(0, len(frame), 32)]))


def check(actual, expected, atol=1e-8, rtol=1e-11):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol)


def save(frame, folder, name):
    frame.to_csv(folder / name, index=False)


def biochar(settings, output):
    cfg = settings['biochar']; frame = read('biochar_records.csv', inputs=True)
    order = np.random.RandomState(cfg['split_seed']).permutation(len(frame))
    cut = int(len(frame) * (1 - cfg['test_fraction']))
    train, test = order[:cut], order[cut:]
    with np.load(ROOT / 'data/biochar_weights.npz', allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    predicted = predict(frame, cfg, weights)
    expected = read('author_checkpoint_predictions.csv').set_index('source_excel_row')
    check(np.log(predicted), expected.loc[frame.index + 2, 'numpy_predicted_log_q'], atol=1e-11)
    framework_log = expected.loc[frame.index + 2, 'predicted_log_q'].to_numpy()
    framework_q = expected.loc[frame.index + 2, 'predicted_q_mg_g'].to_numpy()
    framework_difference = float(np.max(np.abs(np.log(predicted) - framework_log)))
    assert framework_difference < 1e-4
    features = cfg['numeric_features'] + cfg['categorical_features']
    fields = [name for name in features if name != 'loading (g)']
    tuples = [tuple(row) for row in frame[fields].to_numpy()]
    contexts_index, _ = pd.factorize(pd.Series(tuples, dtype=object), sort=False)
    contexts = frame[fields].drop_duplicates().reset_index(drop=True)
    masses, mass_index = np.unique(frame['loading (g)'].to_numpy(float), return_inverse=True)
    frequency = np.zeros((len(contexts), len(masses)))
    np.add.at(frequency, (contexts_index[train], mass_index[train]), 1 / len(train))
    q_grid = np.empty_like(frequency)
    for start in range(0, len(contexts), 16):
        batch = contexts.iloc[start:start+16]
        inputs = batch.loc[batch.index.repeat(len(masses))].reset_index(drop=True)
        inputs['loading (g)'] = np.tile(masses, len(batch))
        q_grid[start:start+len(batch)] = predict(inputs, cfg, weights).reshape(len(batch), -1)
    check(q_grid[contexts_index, mass_index], predicted)
    removal = p.removal_grid(q_grid, contexts, masses)
    game = {name: p.grouped_contributions(grid, frequency, contexts_index[test], mass_index[test])
            for name, grid in [('uptake', q_grid), ('removal', removal)]}
    original = read('author_output_game_records.csv')
    for name, column in [('uptake', 'uptake_loading_phi_mg_g'), ('removal', 'removal_loading_phi_percentage_points')]:
        check(game[name]['mass_phi'], original[column])
    changed = p.sign(game['uptake']['mass_phi']) != p.sign(game['removal']['mass_phi'])
    saved_shap = read('biochar_saved_shap.csv', inputs=True)
    active = ['Ci', 'Volume (L)', 'loading (g)']
    ceiling_phi, _ = p.exact_attributions(frame.iloc[train][active].to_numpy(), frame.iloc[test][active].to_numpy())
    ceiling_expected = read('author_balance_attribution_records.csv')
    for index, field in enumerate(active):
        check(ceiling_phi[:, index], ceiling_expected['ceiling_exact_shap:' + field])
    signs_equal = np.sign(ceiling_phi[:, 2]) == np.sign(saved_shap['loading (g)'])
    solute_mass = (contexts.Ci * contexts['Volume (L)']).to_numpy()
    log_q, log_r, normalizer = p.log_outputs(q_grid, solute_mass, masses)
    log_games = [p.grouped_contributions(grid, frequency, contexts_index[test], mass_index[test])
                 for grid in (log_q, log_r, normalizer)]
    check(log_games[0]['mass_phi'], log_games[1]['mass_phi'] + log_games[2]['mass_phi'], atol=1e-11)
    check(log_games[2]['mass_phi'], p.analytic_mass_normalizer(masses, frequency, mass_index[test]), atol=1e-11)
    log_expected = read('author_log_balance_records.csv')
    check(log_games[1]['mass_phi'], log_expected.log_removal_loading_contribution)
    selected = frame.iloc[test]
    treatment = p.balance_fields(selected.Ci, selected.Cf, selected['loading (g)'],
                                 selected['Volume (L)'], selected.qe, framework_q[test])
    treatment['predicted_log_q'] = framework_log[test]
    subsets = [('all_original_test_records', treatment),
               ('physical_observations', treatment[treatment.physical_observation])]
    subsets += [(f'observed_removal_ge_{int(100 * threshold)}pct',
                 treatment[treatment.physical_observation & treatment.observed_removal_fraction.ge(threshold)])
                for threshold in (.9, .95, .99)]
    precision = pd.DataFrame([p.summary(group, 'original_checkpoint', name) for name, group in subsets])
    reference_precision = read('author_treatment_summary.csv')
    for _, row in precision.iterrows():
        expected_row = reference_precision[(reference_precision.method == 'original_checkpoint')
                                           & (reference_precision.population == row.population)].iloc[0]
        for column in ('records', 'q_r2', 'ct_r2', 'ct_mae_mg_l', 'median_q_relative_error_percent',
                       'median_ct_relative_error_percent', 'nonphysical_ct_predictions'):
            check(row[column], expected_row[column], atol=1e-7)
    save(precision, output, 'biochar_precision.csv')
    save(pd.DataFrame({'source_excel_row': test + 2, 'model_q': predicted[test],
                       'model_uptake_phi': game['uptake']['mass_phi'],
                       'model_removal_phi': game['removal']['mass_phi'], 'ceiling_loading_phi': ceiling_phi[:, 2],
                       'saved_loading_shap': saved_shap['loading (g)'], 'output_sign_changed': changed}),
         output, 'biochar_attributions.csv')
    # The metadata-selected pair and its concentration limits are unchanged.
    endpoints = []
    for index in (38, 39):
        row = frame.iloc[index]; dose = row['loading (g)'] / row['Volume (L)']
        def query(c0, row=row):
            values = np.atleast_1d(np.asarray(c0, float))
            inputs = pd.DataFrame([row.to_dict()] * len(values)); inputs['Ci'] = values
            return predict(inputs, cfg, weights)
        scan = p.scan_response(query, dose, (1., 10.))
        for target in (1., 2.5, 5., 7.5):
            result = p.solve_target(query, target, dose, scan)
            endpoints.append(dict(source_excel_row=index+2, loading_g=float(row['loading (g)']),
                                  target_ct_mg_l=target, **result))
    endpoint_frame = pd.DataFrame(endpoints)
    prior_endpoints = read('author_query_endpoints.csv')
    for _, row in endpoint_frame.iterrows():
        match = prior_endpoints[np.isclose(prior_endpoints.loading_g, row.loading_g)
                                & np.isclose(prior_endpoints.target_ct_mg_l, row.target_ct_mg_l)].iloc[0]
        check(row.c0_mg_l, match.c0_mg_l, atol=1e-7)
        check(row.predicted_q_mg_g, match.predicted_q_mg_g, atol=1e-7)
        assert row.status == match.status
    save(endpoint_frame, output, 'biochar_common_concentration_queries.csv')
    return {'source_records': len(frame), 'training_records': len(train), 'test_records': len(test),
            'uptake_r2': float(r2_score(selected.qe, predicted[test])),
            'source_framework_uptake_r2': float(r2_score(selected.qe, framework_q[test])),
            'numpy_vs_source_framework_maximum_log_difference': framework_difference,
            'precision_statistics_basis': 'Retained source-framework predictions; the independent NumPy replay is checked separately',
            'reference_sign_agreement': int(signs_equal.sum()), 'output_sign_changes': int(changed.sum()),
            'negative_log_uptake_positive_log_removal': int(((p.sign(log_games[0]['mass_phi']) == -1)
                                                           & (p.sign(log_games[1]['mass_phi']) == 1)).sum()),
            'grid_queries': int(q_grid.size), 'nonphysical_hybrid_predictions': int(((removal < 0) | (removal > 100)).sum()),
            'high_removal_subsets': precision[precision.population.str.contains('ge_')].records.astype(int).tolist(),
            'common_concentration_endpoints': len(endpoint_frame)}


def carbon(settings, output):
    cfg = settings['activated_carbon']; frame = read('activated_carbon_records.csv', inputs=True)
    x = frame[cfg['features']].to_numpy(float); y = frame['Adsorption capacity (mg/g)'].to_numpy(float)
    bins = pd.qcut(y, q=cfg['stratification_quantiles'], labels=False, duplicates='drop')
    train, temporary = train_test_split(np.arange(len(frame)), test_size=.3, random_state=cfg['split_seed'], stratify=bins)
    validation, test = train_test_split(temporary, test_size=.5, random_state=cfg['split_seed'], stratify=bins[temporary])
    model = make_pipeline(StandardScaler(), GradientBoostingRegressor(**cfg['parameters']))
    model.fit(x[train], y[train]); predicted = model.predict(x)
    expected = read('activated_carbon_output_game_records.csv')
    check(predicted[test], expected.predicted_q_mg_g, atol=1e-8)
    contexts, context_ids = np.unique(x[:, 1:], axis=0, return_inverse=True)
    doses, dose_ids = np.unique(x[:, 0], return_inverse=True)
    frequency = np.zeros((len(contexts), len(doses)))
    np.add.at(frequency, (context_ids[train], dose_ids[train]), 1 / len(train))
    grid = np.empty_like(frequency)
    for start in range(0, len(contexts), 64):
        batch = contexts[start:start+64]
        inputs = np.column_stack([np.tile(doses, len(batch)), np.repeat(batch, len(doses), axis=0)])
        grid[start:start+len(batch)] = model.predict(inputs).reshape(len(batch), -1)
    removal = 100 * grid * doses[None, :] / contexts[:, 0, None]
    game = {name: p.grouped_contributions(values, frequency, context_ids[test], dose_ids[test])
            for name, values in [('uptake', grid), ('removal', removal)]}
    check(game['uptake']['mass_phi'], expected.uptake_dose_phi_mg_g)
    check(game['removal']['mass_phi'], expected.removal_dose_phi_percentage_points)
    coalitions = p.analytic_coalitions(x[test, 0], x[test, 1], x[train, 0], x[train, 1])
    reference = read('activated_carbon_reference_records.csv')
    scores = {}
    for name, values in coalitions.items():
        phi, _ = p.contributions(values)
        check(phi, reference[name + '_dose_phi_mg_g'])
        scores[name] = {'r2': float(r2_score(y[test], values[:, -1])),
                        'sign_agreement': int((p.sign(phi) == p.sign(game['uptake']['mass_phi'])).sum())}
    pairs, members, fields = p.observed_pairs(frame)
    comparisons = pd.DataFrame(pairs); prior = read('activated_carbon_dose_pairs_records.csv')
    assert len(fields) == 18 and len(comparisons) == len(prior)
    for field in ('low_observed_q_mg_g', 'high_observed_q_mg_g', 'observed_q_direction',
                  'observed_removal_direction', 'observed_removal_delta_percentage_points'):
        check(comparisons[field], prior[field])
    save(comparisons, output, 'activated_carbon_dose_pairs.csv')
    save(pd.DataFrame({'source_excel_row': test + 2, 'observed_q': y[test], 'model_q': predicted[test],
                       'uptake_dose_phi': game['uptake']['mass_phi'], 'removal_dose_phi': game['removal']['mass_phi']}),
         output, 'activated_carbon_attributions.csv')
    counts = {f'{a},{b}': int(((comparisons.observed_q_direction == a)
                              & (comparisons.observed_removal_direction == b)).sum())
              for a, b in [(-1, 1), (-1, -1), (1, 1)]}
    return {'source_records': len(frame), 'training_records': len(train), 'validation_records': len(validation),
            'test_records': len(test), 'uptake_r2': float(r2_score(y[test], predicted[test])),
            'uptake_mae': float(mean_absolute_error(y[test], predicted[test])), 'references': scores,
            'output_sign_changes': int((p.sign(game['uptake']['mass_phi']) != p.sign(game['removal']['mass_phi'])).sum()),
            'matched_series': int(comparisons.series_id.nunique()), 'matched_records': len(members),
            'dose_pairs': len(comparisons), 'reported_source_labels': int(comparisons.reported_reference_label.nunique()),
            'dose_direction_counts': counts, 'nonphysical_hybrid_predictions': int(((removal < 0) | (removal > 100)).sum())}


def controls(output):
    rows = []
    doses = np.array([1., 2.])
    for alpha in (-.05, -.02, 0., .02, .05, .1):
        state = p.solve(np.column_stack([np.full(2, 100.), doses]), alpha)
        common = p.relation(30., doses, alpha)
        check(state['balance_error'], [0., 0.], atol=1e-10)
        common_c0 = 30. + doses * common
        check(p.solve(np.column_stack([common_c0, doses]), alpha)['ce'], [30., 30.], atol=1e-10)
        rows.append({'alpha': alpha, 'fixed_initial_q_change_percent': 100 * (state['q'][1] / state['q'][0] - 1),
                     'common_residual_q_change_percent': 100 * (common[1] / common[0] - 1)})
    frame = pd.DataFrame(rows)
    assert frame.fixed_initial_q_change_percent.lt(0).all()
    check(frame.common_residual_q_change_percent, [5.555555555555555, 2.083333333333333,
                                                   0., -1.923076923076923, -4.545454545454545, -8.333333333333333])
    save(frame, output, 'equilibrium_controls.csv')
    return {'fixed_initial_decreases': len(frame), 'common_residual_directions': ['increase', 'unchanged', 'decrease']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced')
    args = parser.parse_args(); output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory; saved outputs are never overwritten.')
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    settings = json.loads((ROOT / 'data/model_settings.json').read_text())
    started = time.monotonic()
    print('Replaying the released biochar checkpoint and its fixed-background comparisons...', flush=True)
    report = {'biochar': biochar(settings, output)}
    print('Refitting the reported activated-carbon settings and recomputing all matched dose pairs...', flush=True)
    report['activated_carbon'] = carbon(settings, output)
    report['controls'] = controls(output)
    report['elapsed_seconds'] = time.monotonic() - started
    report['reference_comparisons_passed'] = True
    report['scope'] = 'Primary model calculations, source comparisons, and six equilibrium controls; supplementary saved results are supplied separately.'
    (output / 'reproduction_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
