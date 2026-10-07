"""Draw the separate interpretation draft from unchanged, saved evidence."""

from pathlib import Path
import hashlib
import json
import os

PROJECT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parents[1]
CACHE = PROJECT / '.cache'
os.umask(0o077)
CACHE.mkdir(parents=True, mode=0o700, exist_ok=True)
for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'MPLCONFIGDIR'):
    os.environ[key] = str(CACHE)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pypdf import PdfReader
from pypdf.generic import ContentStream

DATA = PROJECT / 'results'
AUDIT = HERE / 'results/review_audit'
FIGURES = HERE / 'figures'
FIGURES.mkdir(mode=0o700, exist_ok=True)
BLUE, ORANGE, RED, GRAY = '#42749C', '#C9904E', '#B75B63', '#737C85'
PALE, DARK = '#DDE3E8', '#28333D'
plt.rcParams.update({
    'font.family': 'Arial', 'font.weight': 'bold', 'axes.labelweight': 'bold',
    'axes.titleweight': 'bold', 'font.size': 10.3, 'axes.titlesize': 11.4,
    'axes.labelsize': 10.5, 'xtick.labelsize': 10.0, 'ytick.labelsize': 10.0,
    'legend.fontsize': 10.0, 'legend.frameon': False, 'axes.grid': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial:bold',
    'mathtext.it': 'Arial:italic:bold', 'mathtext.bf': 'Arial:bold',
    'mathtext.cal': 'Arial:bold',
})

used = {}
audit_used = {}
export_checks = {}


def read(name):
    path = DATA / name
    used[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def read_audit(name):
    path = AUDIT / name
    audit_used[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def panels(count=3, height=3.3):
    fig, axes = plt.subplots(1, count, figsize=(7.2, height), squeeze=False)
    fig.subplots_adjust(left=.14, right=.985, bottom=.29, top=.78,
                        wspace=.88 if count == 3 else .55)
    axes = axes[0]
    for i, ax in enumerate(axes):
        ax.text(-.22 if count == 3 else -.15, 1.20, chr(65 + i),
                transform=ax.transAxes, ha='left', va='bottom', fontsize=14)
        ax.tick_params(which='major', length=4, width=.8, pad=3)
        ax.tick_params(which='minor', length=2.2, width=.6)
    return fig, axes


def identity(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], color=GRAY, linewidth=.9, linestyle='--')


def legend(ax, handles, labels, **kwargs):
    return ax.legend(handles, labels, frameon=False, handlelength=1.2,
                     labelspacing=.3, borderaxespad=.2, **kwargs)


def model_handles():
    return [Line2D([], [], color=BLUE, marker='o', linestyle='none', markersize=5),
            Line2D([], [], color=ORANGE, marker='s', markerfacecolor='white',
                   linestyle='none', markersize=5)]


def save(fig, name):
    # Bounds are explicit; masking is unnecessary and makes Illustrator editing harder.
    for item in fig.findobj():
        if hasattr(item, 'set_clip_on'):
            item.set_clip_on(False)
        if hasattr(item, 'set_clip_path'):
            item.set_clip_path(None)
    # Repeat after tick materialization so labels retain proper font glyphs.
    for _ in range(2):
        for item in fig.findobj(matplotlib.text.Text):
            value = item.get_text().replace('mg L⁻¹', 'mg/L').replace('mg g⁻¹', 'mg/g').replace('⁻¹', '$^{-1}$')
            for before, after in [('C₀', 'C$_0$'), ('Cₑ', 'C$_e$'), ('Cₜ', 'C$_t$'), ('R²', '$R^2$')]:
                value = value.replace(before, after)
            item.set_text(value)
        fig.canvas.draw()
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(FIGURES / f'{name}.{suffix}', dpi=400)
    svg = (FIGURES / f'{name}.svg').read_text()
    assert all(s not in svg for s in ('clipPath', 'clip-path', '<mask', 'mask='))
    assert 'mg L' not in svg
    assert 'mg g' not in svg
    pdf = PdfReader(FIGURES / f'{name}.pdf')
    clips = 0
    for page in pdf.pages:
        clips += sum(op in (b'W', b'W*') for _, op in ContentStream(page.get_contents(), pdf).operations)
    assert clips == 0, (name, clips)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    visible_text = list(fig.texts)
    for ax in fig.axes:
        if ax.title.get_text():
            title_box = ax.title.get_window_extent(renderer)
            for label in ax.texts:
                if len(label.get_text()) == 1 and label.get_text().isupper():
                    assert not title_box.overlaps(label.get_window_extent(renderer)), (
                        name, label.get_text(), 'Panel label overlaps title')
        visible_text += list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend() is not None:
            visible_text += list(ax.get_legend().get_texts())
        for axis, bounds in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if min(bounds) <= tick.get_loc() <= max(bounds):
                    visible_text.extend([tick.label1, tick.label2])
    for item in visible_text:
        if item.get_visible() and item.get_text():
            box = item.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
                outside.append(item.get_text())
    assert not outside, (name, outside)
    export_checks[name] = {'pdf_clipping_operators': clips,
                          'svg_clip_or_mask_constructs': 0,
                          'out_of_canvas_text': outside,
                          'size_inches': list(fig.get_size_inches()),
                          'exports': ['png', 'pdf', 'svg']}
    fig.savefig(CACHE / f'{name}_preview.png', dpi=140)
    plt.close(fig)


def scatter_prediction(ax, observed, model, ceiling, log=False):
    ax.scatter(observed, ceiling, s=18, marker='s', facecolors='none',
               edgecolors=ORANGE, linewidths=.6, zorder=2)
    ax.scatter(observed, model, s=13, color=BLUE, alpha=.7, linewidths=0, zorder=3)
    if log:
        assert all(np.isfinite(v).all() and (v > 0).all() for v in (observed, model, ceiling)), \
            'Logarithmic prediction plots must not silently omit nonpositive values'
        lo = min(observed.min(), model.min(), ceiling.min()) / 1.7
        hi = max(observed.max(), model.max(), ceiling.max()) * 1.7
        ax.set_xscale('log'); ax.set_yscale('log')
    else:
        lo = 0.
        hi = max(observed.max(), model.max(), ceiling.max()) * 1.13
        lo -= .04 * hi
    identity(ax, lo, hi)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Observed uptake\n(mg g⁻¹)')
    ax.set_ylabel('Predicted uptake\n(mg g⁻¹)')


def output_scatter(ax, frame, xcol, ycol, physical_col):
    physical = frame[physical_col].astype(bool)
    for changed, color in ((False, GRAY), (True, BLUE)):
        selected = frame[physical & frame.sign_changed.eq(changed)]
        ax.scatter(selected[xcol], selected[ycol], s=16, color=color,
                   alpha=.75, linewidths=0)
    invalid = frame[~physical]
    ax.scatter(invalid[xcol], invalid[ycol], s=28, color=RED, marker='x',
               linewidths=.9, zorder=4)
    ax.axhline(0, color=GRAY, linewidth=.8, linestyle='--')
    ax.axvline(0, color=GRAY, linewidth=.8, linestyle='--')
    for axis, values in ((ax.xaxis, frame[xcol]), (ax.yaxis, frame[ycol])):
        lo, hi = min(0, values.min()), max(0, values.max())
        pad = max(1, .14 * (hi - lo))
        if axis is ax.xaxis:
            ax.set_xlim(lo - pad, hi + pad)
        else:
            ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('Uptake contribution\n(mg g⁻¹)')
    ax.set_ylabel('Removal contribution\n(percentage points)')


def main_one(a, summaries, output_game, ac_reference, ac_game):
    from sklearn.metrics import r2_score
    fig, grid = plt.subplots(3, 2, figsize=(7.2, 7.7))
    fig.subplots_adjust(left=.145, right=.975, bottom=.17, top=.91,
                        wspace=.73, hspace=1.05)
    axes = grid.ravel()
    for index, ax in enumerate(axes):
        ax.text(-.20, 1.20, chr(65 + index), transform=ax.transAxes,
                ha='left', va='bottom', fontsize=14)
        ax.tick_params(which='major', length=4, width=.8, pad=3)
        ax.tick_params(which='minor', length=2.2, width=.6)
    for ax, frame, ccol, title in zip(axes[:2], (a, ac_reference),
                                    ('ceiling_q_mg_g', 'ceiling_predicted_q_mg_g'),
                                    ('Biochar: prediction', 'Activated carbon: prediction')):
        scatter_prediction(ax, frame.observed_q_mg_g, frame.predicted_q_mg_g,
                           frame[ccol], log=True)
        ax.set_title(title, pad=13, fontsize=10.3)
        legend(ax, model_handles(), ['Model', 'Ceiling'], loc='upper left', fontsize=10)
        model_r2 = r2_score(frame.observed_q_mg_g, frame.predicted_q_mg_g)
        ceiling_r2 = r2_score(frame.observed_q_mg_g, frame[ccol])
        ax.text(.97, .04, f'Raw R²\nModel: {model_r2:.3f}\nCeiling: {ceiling_r2:.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10)
    for ax, x, y, title in (
        (axes[2], a['ceiling_exact_shap:loading (g)'], a['original_saved_shap:loading (g)'],
         'Biochar: saved SHAP'),
        (axes[3], ac_reference.ceiling_dose_phi_mg_g, ac_reference.uptake_dose_phi_mg_g,
         'Activated carbon: two groups')):
        ax.scatter(x, y, s=16, color=BLUE, alpha=.7, linewidths=0)
        ax.axhline(0, color=GRAY, linewidth=.8, linestyle='--')
        ax.axvline(0, color=GRAY, linewidth=.8, linestyle='--')
        for setter, values, upper_pad in ((ax.set_xlim, x, .15), (ax.set_ylim, y, .35)):
            span = max(values.max() - values.min(), 1)
            setter(values.min() - .1 * span, values.max() + upper_pad * span)
        ax.set_title(title, pad=13, fontsize=10.3)
        ax.set_xlabel('Ceiling contribution\n(mg g⁻¹)')
        ax.set_ylabel('Model contribution\n(mg g⁻¹)')
        count = int(np.equal(np.sign(x), np.sign(y)).sum())
        ax.text(.96, .95, f'Signs agree\n{count}/{len(x)}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10)
    output_scatter(axes[4], output_game, 'uptake_loading_phi_mg_g',
                   'removal_loading_phi_percentage_points', 'prediction_in_removal_bounds')
    output_scatter(axes[5], ac_game, 'uptake_dose_phi_mg_g',
                   'removal_dose_phi_percentage_points', 'physical_prediction')
    axes[4].set_title('Biochar: uptake vs removal', pad=13, fontsize=10.3)
    axes[5].set_title('Activated carbon: outputs', pad=13, fontsize=10.3)
    for ax in axes[4:]:
        ax.set_xscale('symlog', linthresh=10)
    fig.legend([Line2D([], [], color=c, marker=m, ls='none', markersize=5)
                for c, m in ((BLUE, 'o'), (GRAY, 'o'), (RED, 'x'))],
               ['Opposite signs (E–F)', 'Matching signs (E–F)', 'Nonphysical output (E–F)'],
               loc='lower center', bbox_to_anchor=(.52, .003), ncol=3,
               frameon=False, fontsize=10, columnspacing=1.0)
    save(fig, 'Figure1')


def main_two(paths, queries, controls):
    fig, axes = panels(height=3.55)
    cases = [(-.05, BLUE, 'Increases'), (0., GRAY, 'Unchanged'), (.1, RED, 'Decreases')]
    for alpha, color, label in cases:
        s = paths[np.isclose(paths.alpha_l_g, alpha) & paths.c0_mg_l.eq(100)
                  & paths.dose_g_l.between(1, 2)]
        axes[0].plot(s.dose_g_l, s.q, color=color, linewidth=1.7)
        q = queries[np.isclose(queries.alpha_l_g, alpha) & queries.target_ce_mg_l.eq(30)].iloc[0]
        axes[1].plot([1, 2], [q.low_q_mg_g, q.high_q_mg_g], color=color,
                     linewidth=1.7, marker='o', markersize=5, label=label)
    for ax in axes[:2]:
        ax.set_xlim(.92, 2.08); ax.set_xticks([1, 1.5, 2]); ax.set_ylim(42, 90)
        ax.set_xlabel('Dose (g L⁻¹)'); ax.set_ylabel('Uptake (mg g⁻¹)')
    axes[0].set_title('Fixed initial\nconcentration', pad=13)
    axes[0].text(.04, .95, 'C₀ = 100 mg L⁻¹', transform=axes[0].transAxes,
                 va='top', fontsize=9.8)
    axes[1].set_title('Common equilibrium\nconcentration', pad=13, fontsize=10.2)
    axes[1].text(.04, .95, 'Cₑ = 30 mg L⁻¹', transform=axes[1].transAxes,
                 va='top', fontsize=9.8)
    legend(axes[1], [Line2D([], [], color=c, lw=1.7) for _, c, _ in cases],
           [label for _, _, label in cases], loc='lower left', fontsize=9.8)
    ax = axes[2]
    selected = controls[controls.dose_g_l.eq(2)].sort_values('alpha_l_g')
    assert len(selected) == 6 and selected.dose_phi.lt(0).all()
    yy = np.arange(len(selected))
    colors = [BLUE if a < 0 else RED if a > 0 else GRAY for a in selected.alpha_l_g]
    ax.hlines(yy, selected.dose_phi, 0, color=PALE, linewidth=2)
    ax.scatter(selected.dose_phi, yy, c=colors, s=40, zorder=3)
    ax.axvline(0, color=GRAY, linewidth=.8, linestyle='--')
    ax.set_xlim(-27, 2); ax.set_ylim(5.5, -.6)
    ax.set_xticks([-25, -15, -5, 0])
    ax.set_yticks(yy, [f'{a:g}' for a in selected.alpha_l_g])
    ax.set_title('Exact dose\ncontributions', pad=13)
    ax.set_xlabel('Dose contribution\n(mg g⁻¹)')
    ax.set_ylabel('Capacity parameter α (L/g)')
    save(fig, 'Figure2')


def main_three(pairs, records):
    a = records[records.method.eq('original_checkpoint') & records.physical_observation
                & records.observed_removal_fraction.ge(.9)].copy()
    assert len(a) == 44
    fig, axes = panels(height=3.6)
    ax = axes[0]
    colors = [BLUE, RED, ORANGE]
    keys = [(-1, 1), (-1, -1), (1, 1)]
    labels = ['q ↓ / removal ↑', 'Both ↓', 'Both ↑']
    for (qs, rs), color in zip(keys, colors):
        subset = pairs[pairs.observed_q_direction.eq(qs) & pairs.observed_removal_direction.eq(rs)]
        ax.scatter(100 * subset.observed_q_relative_change,
                   subset.observed_removal_delta_percentage_points,
                   color=color, s=24, alpha=.8, linewidths=0)
    ax.axhline(0, color=GRAY, lw=.8, ls='--')
    xv, yv = 100 * pairs.observed_q_relative_change, pairs.observed_removal_delta_percentage_points
    ax.plot([0, 0], [yv.min()-8, yv.max()+8], color=GRAY, lw=.8, ls='--')
    ax.set_xlim(xv.min()-10, xv.max()+10)
    ax.set_ylim(yv.min()-8, yv.max()+42)
    ax.set_xlabel('Uptake change (%)')
    ax.set_ylabel('Removal change\n(percentage points)')
    ax.set_title('Recorded dose\nchanges', pad=13)
    legend(ax, [Line2D([], [], color=c, marker='o', ls='', markersize=4) for c in colors],
           labels, loc='upper left', fontsize=8.6)
    ax = axes[1]
    ax.scatter(a.observed_ct_mg_l, np.zeros(len(a)), s=22, marker='s',
               facecolors='none', edgecolors=ORANGE, linewidths=.7)
    ax.scatter(a.observed_ct_mg_l, a.predicted_ct_mg_l, s=22, color=BLUE,
               alpha=.8, linewidths=0)
    lo = min(0, a.predicted_ct_mg_l.min()) - .09
    hi = max(a.observed_ct_mg_l.max(), a.predicted_ct_mg_l.max()) * 1.1
    identity(ax, 0, hi)
    ax.axhline(0, color=GRAY, linewidth=.8, linestyle=':')
    ax.set_xlim(-.04, hi); ax.set_ylim(lo, hi * 1.3)
    ax.set_title('Residual\nconcentration', pad=13)
    ax.set_xlabel('Reference Cₜ (mg L⁻¹)'); ax.set_ylabel('Predicted Cₜ (mg L⁻¹)')
    legend(ax, model_handles(), ['Model', 'Ceiling'],
           loc='upper left', fontsize=9.8)
    ax = axes[2]
    qerr = 100 * a.q_absolute_relative_error.to_numpy()
    cerr = 100 * a.ct_absolute_relative_error.to_numpy()
    assert min(qerr.min(), cerr.min()) > 0
    jitter = np.linspace(-.11, .11, len(a))
    for i, (q, c) in enumerate(zip(qerr, cerr)):
        ax.plot([jitter[i], 1 + jitter[i]], [q, c], color=PALE, linewidth=.6, zorder=1)
    ax.scatter(jitter, qerr, color=BLUE, s=19, zorder=2)
    ax.scatter(1 + jitter, cerr, color=RED, s=19, zorder=2)
    for x, v in [(0, qerr), (1, cerr)]:
        median = np.median(v)
        ax.plot([x - .17, x + .17], [median, median], color=DARK, linewidth=2.3)
        ax.text(x, max(qerr.max(), cerr.max()) * 2.8, f'{median:.1f}%', ha='center', fontsize=10.2)
    ax.set_yscale('log'); ax.set_ylim(min(qerr.min(), cerr.min()) / 2,
                                    max(qerr.max(), cerr.max()) * 7)
    ax.set_xlim(-.38, 1.38); ax.set_xticks([0, 1], ['Uptake q', 'Residual\nC$_t$'])
    ax.set_title('Paired prediction\nerrors', pad=13)
    ax.set_ylabel('Absolute relative error (%)')
    save(fig, 'Figure3')


def supplement_seven(pairs, sources, records):
    fig = plt.figure(figsize=(7.2, 4.8))
    left = fig.add_axes([.12, .27, .31, .56])
    right = fig.add_axes([.73, .27, .23, .56])
    colors = [BLUE, RED, ORANGE]
    keys = [(-1, 1), (-1, -1), (1, 1)]
    labels = ['Uptake ↓ / removal ↑', 'Both ↓', 'Both ↑']
    for ax, letter in ((left, 'A'), (right, 'B')):
        ax.text(-.17, 1.17, letter, transform=ax.transAxes, fontsize=14,
                ha='left', va='bottom')
        ax.tick_params(which='major', length=4, width=.8, pad=3)
    a = records[records.method.eq('original_checkpoint') & records.physical_observation
                & records.observed_removal_fraction.ge(.9)]
    assert len(a) == 44
    scatter_prediction(left, a.observed_q_mg_g, a.predicted_q_mg_g,
                       a.observed_c0_mg_l/a.dose_g_l, log=True)
    left.set_title('High-removal uptake', pad=13, fontsize=10.6)
    legend(left, model_handles(), ['Model', 'Ceiling'], loc='upper left')
    left.text(.95, .06, 'Raw R²\nModel: 0.9973\nCeiling: 0.9971',
              transform=left.transAxes, ha='right', va='bottom', fontsize=9.6)
    sources = sources.copy()
    yy = np.arange(len(sources))
    start = np.zeros(len(sources))
    for (qs, rs), color in zip(keys, colors):
        counts = np.array([len(pairs[pairs.reported_reference_label.eq(label)
                                    & pairs.observed_q_direction.eq(qs)
                                    & pairs.observed_removal_direction.eq(rs)])
                           for label in sources.reported_reference_label])
        share = 100 * counts / sources.adjacent_pairs.to_numpy()
        right.barh(yy, share, left=start, color=color, height=.66)
        start += share
    np.testing.assert_allclose(start, 100)
    right.set_yticks(yy, sources.reported_reference_label.str.replace(' et al', '', regex=False), fontsize=9.2)
    right.set_ylim(len(sources) - .4, -.8)
    right.set_xlim(0, 117); right.set_xticks([0, 50, 100])
    right.set_xlabel('Comparisons (%)')
    right.set_title('All source labels', pad=13, fontsize=10.6)
    for y, count in zip(yy, sources.adjacent_pairs):
        right.text(104, y, str(count), va='center', fontsize=9.2)
    right.text(104, -1.02, 'n', fontsize=9.2, ha='left')
    legend(right, [Rectangle((0, 0), 1, 1, color=c) for c in colors],
           labels, loc='upper center', bbox_to_anchor=(.5, -.19), fontsize=8.8)
    save(fig, 'FigureS7')


def supplement_eight(saved):
    fig, grid = plt.subplots(2, 2, figsize=(7.2, 6.6))
    fig.subplots_adjust(left=.14, right=.98, bottom=.13, top=.88,
                        wspace=1.0, hspace=.92)
    axes = grid.ravel()
    for i, ax in enumerate(axes):
        ax.text(-.20, 1.22, chr(65+i), transform=ax.transAxes, fontsize=14)
        ax.tick_params(which='major', length=4, width=.8)
        ax.tick_params(which='minor', length=2.2, width=.6)
    ax = axes[0]
    ax.scatter(1000*saved.loading_g, saved['original_saved_shap:loading (g)'],
               color=BLUE, s=17, alpha=.75, linewidths=0)
    ax.axhline(0, color=GRAY, lw=.8, ls='--')
    ax.axvline(6.348655, color=ORANGE, lw=1.2, ls=':')
    ax.set_xscale('log'); ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel('Adsorbent mass (mg)')
    ax.set_ylabel('Saved loading SHAP\n(mg g⁻¹)')
    ax.set_title('Biochar loading gap', pad=15, fontsize=10.4)
    ax.text(.96, .94, 'Reference threshold\n6.35 mg', ha='right', va='top',
            transform=ax.transAxes, color=ORANGE, fontsize=8.6)
    bg = []
    backgrounds = ['training_rows', 'unique_input_profiles', 'balanced', 'same_pollutant']
    labels = ['Training rows', 'Unique inputs', 'Equal labels', 'Same solute']
    for name, color, marker, label in [('biochar', BLUE, 'o', 'Biochar'),
                                        ('activated_carbon', RED, 's', 'Activated carbon')]:
        frame = read_audit(name+'_background_summary.csv')
        frame['background'] = frame.background.replace({
            'material_label_balanced': 'balanced', 'source_label_balanced': 'balanced'})
        frame = frame.set_index('background').loc[backgrounds]
        bg.append((frame, color, marker, label))
    for ax, column, title, xlabel in [
        (axes[1], 'ceiling_sign_agreement', 'Ceiling sign agreement', 'Matching signs (%)'),
        (axes[2], 'sign_reversals', 'Output sign changes', 'Opposite signs (%)')]:
        for j, (frame, color, marker, label) in enumerate(bg):
            ax.plot(100*frame[column]/frame.records, np.arange(4)+(j-.5)*.18,
                    color=color, marker=marker, ms=5, lw=1.2, label=label)
        ax.set_yticks(np.arange(4), labels, fontsize=8.8)
        ax.set_ylim(3.6, -.6); ax.set_xlim(0, 106); ax.set_xticks([0, 50, 100])
        ax.set_xlabel(xlabel); ax.set_title(title, pad=15, fontsize=10.4)
    ax = axes[3]
    for name, color, marker, label in [('biochar', BLUE, 'o', 'Biochar'),
                                      ('activated_carbon', RED, 's', 'Activated carbon')]:
        values = read_audit(name+'_bounded_function_records.csv').groupby('function').sign_reversed.mean()
        ax.plot([0, 1], 100*values.loc[['original', 'bounded_function']],
                color=color, marker=marker, ms=5, lw=1.3)
    ax.set_xticks([0, 1], ['Original', 'Bounded'])
    ax.set_xlim(-.2, 1.2); ax.set_ylim(0, 100)
    ax.set_ylabel('Opposite signs (%)'); ax.set_xlabel('Explained function')
    ax.set_title('Bounding sensitivity', pad=15, fontsize=10.4)
    legend(ax, [Line2D([], [], color=c, marker=m, lw=1.2) for c,m in [(BLUE,'o'),(RED,'s')]],
           ['Biochar', 'Activated carbon'], loc='lower left', fontsize=8.6)
    # The B-D mapping is repeated locally in B to avoid a disconnected legend.
    legend(axes[1], [Line2D([], [], color=c, marker=m, lw=1.2) for c,m in [(BLUE,'o'),(RED,'s')]],
           ['Biochar', 'Activated carbon'], loc='lower left', fontsize=8.6)
    save(fig, 'FigureS8')


def supplement_one(a):
    fig, axes = panels(2, 3.6)
    for ax, key, title in zip(axes, ['Ci', 'Volume (L)'], ['Initial concentration', 'Solution volume']):
        x, y = a[f'ceiling_exact_shap:{key}'], a[f'original_saved_shap:{key}']
        ax.scatter(x, y, color=BLUE, s=18, alpha=.7, linewidths=0)
        ax.axhline(0, color=GRAY, ls='--', lw=.8); ax.axvline(0, color=GRAY, ls='--', lw=.8)
        ax.set_title(title, pad=13)
        ax.set_xlabel('Ceiling Shapley value (mg g⁻¹)')
        ax.set_ylabel('Model SHAP value (mg g⁻¹)')
    save(fig, 'FigureS1')


def supplement_two(original_queries):
    p = read('author_query_operating_pair.csv')
    fig, axes = panels(3, 3.5)
    dose = p.dose_g_l.to_numpy()
    axes[0].plot(dose, p.observed_q_mg_g, color=DARK, marker='o', lw=1.7)
    axes[0].plot(dose, p.predicted_q_mg_g, color=BLUE, marker='s', lw=1.7)
    axes[0].set_ylabel('Uptake (mg g⁻¹)'); axes[0].set_ylim(.7, 2.8)
    legend(axes[0], [Line2D([], [], color=DARK, marker='o'),
                     Line2D([], [], color=BLUE, marker='s')],
           ['Observed', 'Original model'], loc='lower left')
    axes[1].plot(dose, 100 * dose * p.observed_q_mg_g / 10,
                 color=DARK, marker='o', lw=1.7)
    axes[1].plot(dose, 100 * dose * p.predicted_q_mg_g / 10,
                 color=BLUE, marker='s', lw=1.7)
    axes[1].set_ylabel('Removal (%)'); axes[1].set_ylim(10, 40)
    legend(axes[1], [Line2D([], [], color=DARK, marker='o'),
                     Line2D([], [], color=BLUE, marker='s')],
           ['Observed', 'Original model'], loc='upper left')
    for ax, title in zip(axes, ['Uptake per gram', 'Removal (%)']):
        ax.set_title(title, pad=13); ax.set_xlabel('Dose (g L⁻¹)')
        ax.set_xticks([1, 2]); ax.set_xlim(.85, 2.15)
    ax = axes[2]
    ax.plot(original_queries.target_ct_mg_l, original_queries.low_q_mg_g,
            color=BLUE, marker='o', markersize=5, linewidth=1.4)
    ax.plot(original_queries.target_ct_mg_l, original_queries.high_q_mg_g,
            color=BLUE, marker='s', markerfacecolor='white', markersize=5,
            linestyle='--', linewidth=1.4)
    ax.set_xlim(.5, 8); ax.set_ylim(.1, 2.65)
    ax.set_xticks([1, 2.5, 5, 7.5])
    ax.set_title('Common residual\nconcentration', pad=13)
    ax.set_xlabel('Target Cₜ (mg/L)'); ax.set_ylabel('Predicted uptake\n(mg g⁻¹)')
    legend(ax, [Line2D([], [], color=BLUE, marker='o', lw=1.4),
                Line2D([], [], color=BLUE, marker='s', markerfacecolor='white',
                       ls='--', lw=1.4)], ['1 g/L', '2 g/L'], loc='upper left')
    save(fig, 'FigureS2')


def supplement_three(q):
    fig, axes = panels(2, 3.6)
    alphas = sorted(q.alpha_l_g.unique()); targets = sorted(q.target_ce_mg_l.unique())
    ax = axes[0]
    for i, alpha in enumerate(alphas):
        for j, target in enumerate(targets):
            row = q[np.isclose(q.alpha_l_g, alpha) & q.target_ce_mg_l.eq(target)].iloc[0]
            color = BLUE if row.valid_comparison else PALE
            ax.add_patch(Rectangle((j - .45, i - .43), .90, .86,
                                   facecolor=color, edgecolor='white', linewidth=.6))
            ax.text(j, i, 'Yes' if row.valid_comparison else 'No', ha='center', va='center',
                    color='white' if row.valid_comparison else DARK, fontsize=10)
    ax.set_xticks(range(len(targets)), [str(int(t)) for t in targets])
    ax.set_yticks(range(len(alphas)), [f'{a:g}' for a in alphas])
    ax.set_xlim(-.5, 3.5); ax.set_ylim(-.5, 5.5)
    ax.set_title('Roots within tested range', pad=13)
    ax.set_xlabel('Target Cₑ (mg/L)'); ax.set_ylabel('α (L g⁻¹)')
    ax = axes[1]
    s = q[q.target_ce_mg_l.eq(30)]
    ax.plot(s.alpha_l_g, s.saved_fixed_c0_q_change_percent, color=RED, marker='o', lw=1.5)
    ax.plot(s.alpha_l_g, s.estimated_common_ce_change_percent, color=BLUE, marker='s', lw=1.5)
    ax.axhline(0, color=GRAY, lw=.8, ls='--')
    ax.set_ylim(-45, 13); ax.set_xlim(-.065, .115)
    ax.set_xticks([-.05, 0, .05, .10]); ax.set_ylabel('Uptake change (%)')
    ax.set_xlabel('α (L g⁻¹)'); ax.set_title('Two different comparisons', pad=13)
    legend(ax, [Line2D([], [], color=RED, marker='o'), Line2D([], [], color=BLUE, marker='s')],
           ['Fixed C₀ = 100', 'Common Cₑ = 30'], loc='center left')
    save(fig, 'FigureS3')


def supplement_four():
    k = read('real_concave_profiles/kinetic_shared_pairs.csv')
    c = read('concave_equivalence/competitive_contrasts.csv')
    fig, axes = panels(2, 3.6)
    ax = axes[0]
    ax.plot(k.time_h, k.operating_q_change_percent, color=RED, marker='o', lw=1.4)
    ax.plot(k.time_h, k.common_ct_q_change_percent, color=BLUE, marker='s', lw=1.4)
    ax.axhline(0, color=GRAY, lw=.8, ls='--')
    ax.set_xscale('log'); ax.set_xlabel('Contact time (h)'); ax.set_ylim(-60, 110)
    ax.set_ylabel('Uptake change (%)'); ax.set_title('Finite-time response', pad=13)
    legend(ax, [Line2D([], [], color=RED, marker='o'), Line2D([], [], color=BLUE, marker='s')],
           ['Fixed C₀', 'Common Cₜ'],
           loc='upper right', fontsize=9.8)
    ax = axes[1]
    s = c[c.target_a_mmol_l.eq(.1)]
    ax.plot(s.b0_mmol_l, s.common_a_qa_change_percent, color=BLUE, marker='o', lw=1.4)
    ax.axhline(0, color=GRAY, lw=.8, ls='--')
    ax.set_xlim(-.13, 2.13); ax.set_ylim(-1.4, 27)
    ax.set_xticks([0, .5, 1, 2]); ax.set_xlabel('Initial competitor (mmol L⁻¹)')
    ax.set_ylabel('Common-target uptake change (%)')
    ax.set_title('Competitive-solute response', pad=13)
    ax.text(.04, .95, 'Common analyte residual:\n0.1 mmol L⁻¹',
            transform=ax.transAxes, va='top', fontsize=9.8)
    save(fig, 'FigureS5')


def supplement_five():
    d = read('real_concave_profiles/zhao_full_markers.csv')
    fig, axes = panels(2, 3.7)
    colors = [BLUE, ORANGE, GRAY, RED]
    markers = ['o', 's', '^', 'D']
    for ax, metal in zip(axes, ['Pb(II)', 'Cu(II)']):
        for dose, color, marker in zip([1, 2, 4, 8], colors, markers):
            s = d[d.metal.eq(metal) & d.dose_g_l.eq(dose)]
            ax.errorbar(s.ce_mg_l, s.q_mg_g,
                        xerr=np.vstack([s.ce_mg_l - s.ce_low, s.ce_high - s.ce_mg_l]),
                        yerr=np.vstack([s.q_mg_g - s.visible_q_low, s.visible_q_high - s.q_mg_g]),
                        fmt=marker, color=color, markersize=4.5, elinewidth=.7,
                        capsize=2, markerfacecolor='white')
        ax.set_title(metal, pad=13); ax.set_xlabel('Residual concentration (mg L⁻¹)')
        ax.set_ylabel('Uptake (mg g⁻¹)'); ax.set_xlim(-12, 690); ax.set_ylim(-8, 255)
        legend(ax, [Line2D([], [], color=co, marker=ma, markerfacecolor='white', ls='none')
                    for co, ma in zip(colors, markers)],
               [f'{v} g L⁻¹' for v in [1, 2, 4, 8]], loc='upper left', ncol=2,
               columnspacing=.8, fontsize=9.8)
    save(fig, 'FigureS4')


def supplement_six(r):
    a = r[r.method.eq('original_checkpoint') & r.physical_observation]
    fig, axes = panels(2, 3.6)
    ax = axes[0]
    ax.scatter(a.observed_removal_fraction * 100, a.q_absolute_relative_error * 100,
               color=BLUE, s=14, alpha=.65, linewidths=0)
    ax.scatter(a.observed_removal_fraction * 100, a.ct_absolute_relative_error * 100,
               color=RED, s=14, alpha=.65, linewidths=0)
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlim(-2, 102); ax.set_xlabel('Record-derived removal (%)')
    ax.set_ylabel('Absolute relative error (%)'); ax.set_title('All physical observations', pad=13)
    legend(ax, [Line2D([], [], color=BLUE, marker='o', ls='none'),
                Line2D([], [], color=RED, marker='o', ls='none')],
           ['Uptake', 'Residual concentration'], loc='upper left', fontsize=9.8)
    ax = axes[1]
    ax.scatter(a.observed_ct_mg_l, a.predicted_ct_mg_l,
               color=BLUE, s=14, alpha=.65, linewidths=0)
    lo = min(a.predicted_ct_mg_l.min(), 0) - .2
    hi = max(a.observed_ct_mg_l.max(), a.predicted_ct_mg_l.max()) * 1.08
    identity(ax, 0, hi); ax.axhline(0, color=GRAY, lw=.8, ls=':')
    ax.set_xlim(-.2, hi); ax.set_ylim(lo, hi)
    ax.set_title('Residual prediction', pad=13)
    ax.set_xlabel('Reference Cₜ (mg L⁻¹)'); ax.set_ylabel('Predicted Cₜ (mg L⁻¹)')
    save(fig, 'FigureS6')


def main():
    a = read('author_balance_attribution_records.csv')
    s = read('author_balance_attribution_summary.csv')
    r = read('author_treatment_records.csv')
    q = read('concave_equivalence/surface_query_comparisons.csv')
    paths = read('concave_equivalence/simple_law_paths.csv')
    original_queries = read('author_query_comparisons.csv')
    output_game = read('author_output_game_records.csv')
    ac_reference = read('activated_carbon_reference_records.csv')
    ac_game = read('activated_carbon_output_game_records.csv')
    dose_pairs = read('activated_carbon_dose_pairs_records.csv')
    dose_sources = read('activated_carbon_dose_pairs_sources.csv')
    assert len(a) == 456
    assert np.array_equal(np.sign(a['original_saved_shap:loading (g)']),
                          np.sign(a['ceiling_exact_shap:loading (g)']))
    assert len(output_game) == 456 and output_game.sign_changed.sum() == 353
    assert len(ac_reference) == 230 and ac_reference.ceiling_dose_sign_matches_model.all()
    assert len(ac_game) == 230 and ac_game.sign_changed.sum() == 95
    assert len(dose_pairs) == 158 and len(dose_sources) == 17
    main_one(a, s, output_game, ac_reference, ac_game)
    main_two(paths, q, read_audit('control_attributions.csv'))
    main_three(dose_pairs, r)
    supplement_one(a)
    supplement_two(original_queries)
    supplement_three(q)
    supplement_four()
    supplement_five()
    supplement_six(r)
    supplement_seven(dose_pairs, dose_sources, r)
    supplement_eight(a)
    for name, value in used.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == value
    for name, value in audit_used.items():
        assert hashlib.sha256((AUDIT / name).read_bytes()).hexdigest() == value
    (Path(__file__).parent / 'figure_checks.json').write_text(json.dumps({
        'inputs_unchanged': used, 'review_audit_inputs': audit_used, 'backend': 'Python/matplotlib',
        'figures': export_checks,
    }, indent=2) + '\n')
    print(json.dumps({'figures': list(export_checks), 'source_files_unchanged': len(used)}))


if __name__ == '__main__':
    main()
