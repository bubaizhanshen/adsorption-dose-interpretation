"""Small independent checks of the comparisons used by the primary runner."""

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.umask(0o077)
cache = ROOT / '.cache'
cache.mkdir(mode=0o700, exist_ok=True)
for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME'):
    os.environ[key] = str(cache)
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import physics as p
import auxiliary_physics as a


class PhysicalComparisons(unittest.TestCase):
    def test_monotone_outer_bounds_contain_known_linear_curves(self):
        low = a.MonotoneRectangles(pd.DataFrame(dict(ce_low=[1., 3.], ce_high=[1., 3.],
                                                    q_low=[2., 6.], q_high=[2., 6.])))
        high = a.MonotoneRectangles(pd.DataFrame(dict(ce_low=[1., 3.], ce_high=[1., 3.],
                                                     q_low=[1., 3.], q_high=[1., 3.])))
        bounds = a.component_bounds(low, high, 6., 1., 2.)
        # Both operating concentrations are 2; the actual contrast is -2 mg/g.
        self.assertLessEqual(bounds['direct_lower'], -2.)
        self.assertGreaterEqual(bounds['direct_upper'], -2.)
        self.assertEqual(low.root_bounds(3., 1.), (1., 1.))
        with self.assertRaises(ValueError):
            a.MonotoneRectangles(pd.DataFrame(dict(ce_low=[1., 3.], ce_high=[1., 3.],
                                                    q_low=[6., 2.], q_high=[6., 2.])))

    def test_kinetic_state_initial_and_long_time_limits(self):
        for dose in (.1, 1., 2.):
            initial = a.kinetic_state(100., dose, 0.)
            self.assertEqual(float(initial['q']), 0.)
            self.assertEqual(float(initial['ct']), 100.)
            limit = a.kinetic_state(100., dose, 1e6)
            self.assertAlmostEqual(float(limit['q']), a.HENRY_K*float(limit['ct']))
            self.assertAlmostEqual(float(limit['ct']+dose*limit['q']), 100.)

    def test_competition_holds_complete_solution_state_not_only_analyte(self):
        for target in (.1, .5):
            no_b = [a.direct_at_shared_a(target, 0., d) for d in (1., 2.)]
            self.assertAlmostEqual(no_b[0].qa_mmol_g, no_b[1].qa_mmol_g)
            with_b = [a.direct_at_shared_a(target, 1., d) for d in (1., 2.)]
            self.assertGreater(with_b[1].qa_mmol_g, with_b[0].qa_mmol_g)
            self.assertLess(with_b[1].b_mmol_l, with_b[0].b_mmol_l)
        ca, cb = .1, .25
        qa, qb = ca/(1+ca+cb), cb/(1+ca+cb)
        for dose in (1., 2.):
            state = a.equilibrate(ca+dose*qa, cb+dose*qb, dose)
            np.testing.assert_allclose([state.a_mmol_l, state.b_mmol_l, state.qa_mmol_g, state.qb_mmol_g],
                                       [ca, cb, qa, qb], atol=1e-12)

    def test_ceiling_efficiency(self):
        background = np.array([[10., .1, .1], [80., .2, .4], [30., .3, .2]])
        queries = np.array([[25., .2, .3], [50., .1, .05]])
        phi, values = p.exact_attributions(background, queries)
        np.testing.assert_allclose(phi.sum(axis=1), p.ceiling(queries) - p.ceiling(background).mean())
        np.testing.assert_allclose(values[:, -1], p.ceiling(queries))

    def test_group_game_against_two_orders(self):
        values = np.array([[1., 4.], [5., 12.]])
        weights = np.array([[.4, .1], [.2, .3]])
        contexts = np.array([0, 1]); doses = np.array([1, 0])
        game = p.grouped_contributions(values, weights, contexts, doses)
        baseline = sum(weights[c, d] * values[c, d] for c in range(2) for d in range(2))
        direct = []
        for context, dose in zip(contexts, doses):
            first = sum(weights[c, d] * values[c, dose] for c in range(2) for d in range(2)) - baseline
            last = values[context, dose] - sum(weights[c, d] * values[context, d] for c in range(2) for d in range(2))
            direct.append((first + last) / 2)
        np.testing.assert_allclose(game['mass_phi'], direct)
        np.testing.assert_allclose(game['mass_phi'] + game['context_phi'], game['predicted'] - baseline)
        with self.assertRaises(ValueError):
            p.grouped_contributions(values, -weights, contexts, doses)

    def test_control_against_independent_balance_root(self):
        for alpha in (-.05, 0., .1):
            state = p.solve(np.array([[100., 1.], [100., 2.]]), alpha)
            for position, dose in enumerate((1., 2.)):
                root = brentq(lambda ce: ce + dose * p.relation(ce, dose, alpha) - 100., 0., 100.)
                self.assertAlmostEqual(root, state['ce'][position], places=9)

    def test_matching_does_not_cross_other_conditions(self):
        frame = pd.DataFrame([
            [1, 1., 10., 2., 'Study A', 'BC', 2020, 'Cd', 7.],
            [2, 2., 10., 1.5, 'Study A', 'BC', 2020, 'Cd', 7.],
            [3, 3., 10., 1.2, 'Study A', 'BC', 2020, 'Cd', 8.],
        ], columns=['ID', p.DOSE, p.C0, p.UPTAKE, 'Reference', 'Adsorbent code', 'Year', 'Heavy metal', 'pH'])
        pairs, members, fields = p.observed_pairs(frame)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(members), 2)
        self.assertIn('pH', fields)
        self.assertEqual((pairs[0]['observed_q_direction'], pairs[0]['observed_removal_direction']), (-1, 1))

    def test_residual_error_amplification(self):
        frame = p.balance_fields([10.], [1.], [5.], [1.], [1.8], [1.98])
        np.testing.assert_allclose(frame.ct_absolute_relative_error,
                                   frame.q_absolute_relative_error * frame.error_amplification)
        self.assertAlmostEqual(frame.error_amplification.iloc[0], 9.)

    def test_equilibrium_derivative_identity(self):
        for alpha in (-.05, -.02, 0., .02, .05, .1):
            for c0 in (50., 100., 200.):
                for dose in (.5, 1., 2., 3.):
                    qmax = 100/(1+alpha*dose)
                    ce = brentq(lambda c: c+dose*qmax*.1*c/(1+.1*c)-c0, 0, c0)
                    uptake = qmax*.1*ce/(1+.1*ce)
                    hc = qmax*.1/(1+.1*ce)**2
                    hd = -100*alpha/(1+alpha*dose)**2*.1*ce/(1+.1*ce)
                    expected = (hd-uptake*hc)/(1+dose*hc)
                    step = 1e-5
                    values = p.solve(np.array([[c0, dose-step], [c0, dose+step]]), alpha)['q']
                    np.testing.assert_allclose((values[1]-values[0])/(2*step), expected,
                                               rtol=2e-7, atol=1e-8)

    @unittest.skipUnless((ROOT/'results/review_audit/high_removal_records.csv').is_file(),
                         'Requires the separately obtained empirical record registry')
    def test_review_high_removal_registry_is_not_duplicated_by_method(self):
        records = pd.read_csv(ROOT/'results/review_audit/high_removal_records.csv')
        self.assertEqual(len(records), 44)
        self.assertTrue(records.source_excel_row.is_unique)
        self.assertEqual(set(records.method), {'original_checkpoint'})
        self.assertEqual(int((~records.physical_prediction).sum()), 7)
        self.assertEqual(records.material_label.nunique(), 5)
        self.assertEqual(records.pollutant.nunique(), 6)
        np.testing.assert_allclose(records.ct_absolute_relative_error,
                                   records.q_absolute_relative_error*records.error_amplification)

    @unittest.skipUnless(all((ROOT/f'results/review_audit/{name}_background_{kind}.csv').is_file()
                             for name in ('biochar', 'activated_carbon')
                             for kind in ('records', 'summary')),
                         'Requires the separately obtained empirical background tables')
    def test_background_counts_derive_from_record_level_attributions(self):
        for name in ('biochar', 'activated_carbon'):
            records = pd.read_csv(ROOT/f'results/review_audit/{name}_background_records.csv')
            summary = pd.read_csv(ROOT/f'results/review_audit/{name}_background_summary.csv')
            for row in summary.itertuples(index=False):
                subset = records[records.background.eq(row.background)]
                self.assertTrue(subset.source_excel_row.is_unique)
                self.assertEqual(len(subset), row.records)
                self.assertEqual(int(subset.sign_reversed.sum()), row.sign_reversals)
                self.assertEqual(int(subset.ceiling_sign_agrees.sum()), row.ceiling_sign_agreement)

    @unittest.skipUnless(all((ROOT/f'results/review_audit/{name}_nonphysical_contributions.csv').is_file()
                             for name in ('biochar', 'activated_carbon')),
                         'Requires the separately obtained empirical contribution tables')
    def test_nonphysical_numerical_partition_closes(self):
        for name in ('biochar', 'activated_carbon'):
            records = pd.read_csv(ROOT/f'results/review_audit/{name}_nonphysical_contributions.csv')
            np.testing.assert_allclose(records.full_contribution,
                records.contribution_through_nonphysical_values+records.contribution_through_physical_values,
                rtol=1e-10, atol=1e-9)

    @unittest.skipUnless((ROOT/'results/review_audit/control_attributions.csv').is_file(),
                         'Requires saved manuscript controls; use examples/ for fresh controls')
    def test_all_six_control_attributions_use_recorded_coalitions(self):
        records = pd.read_csv(ROOT/'results/review_audit/control_attributions.csv')
        expected = .5*(records.dose_coalition-records.baseline
                       +records.fixed_c0_uptake-records.c0_coalition)
        np.testing.assert_allclose(records.dose_phi, expected, atol=1e-12)
        selected = records[records.dose_g_l.eq(2)]
        self.assertEqual(len(selected), 6)
        self.assertTrue(selected.dose_phi.lt(0).all())


if __name__ == '__main__':
    unittest.main()
