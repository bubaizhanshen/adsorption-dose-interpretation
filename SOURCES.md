# Sources and input requirements

## Biochar

- Jaffari et al., *Transformer-based deep learning models for adsorption
  capacity prediction of heavy metal ions toward biochar-based adsorbents*.
  https://doi.org/10.1016/j.jhazmat.2023.132773
- Original implementation: https://gitlab.com/atrcheema/envai101
- Inspected revision: `fc6fd3a9b2adcb5c05a733222201c3707b41d907`.
- Architecture dependency, AI4Water:
  `e9d69921c75ad9cd7ce610e5a0e5a3036144d518`.

The primary runner expects prepared `data/biochar_records.csv`,
`data/biochar_saved_shap.csv`, `data/biochar_weights.npz`, and the biochar entry
in `data/model_settings.json`. These files contain records, saved explanations,
weights, numerical-feature names, category vocabularies, and split settings.
They are excluded from this release. The acquisition and preprocessing are
described in Supporting Information Texts S1 and S2. `models.py` is an
independent NumPy implementation of the inspected inference operations, not
the source authors' framework or weights.

The source specifies `nsamples=200` but does not record the installed SHAP
version or effective default regularization. Archived Kernel SHAP values and
new exact two-group contributions are different computations. The independent
near-zero check does not reproduce the historical Kernel SHAP regression.

## Activated carbon

- Lamsiah et al., *Prediction of heavy metal adsorption by activated carbon
  using machine learning*. https://doi.org/10.1016/j.aichem.2026.100117
- Original data and notebook:
  https://github.com/S-Lamsiah/Heavy-Metal-Adsorption-Activated-Carbon-ML
- Inspected revision: `99d7d12a6c60aa9e86f68027c440cd2587d2c9e5`.

The runner refits the reported gradient-boosting settings. It expects the
prepared workbook records and settings under `data/`; see Text S7 for the
split and preprocessing. Reported material and reference labels are retained
as compilation labels, not certified experimental batch identities.

## Dose-dependent isotherms

- Zhao et al., *Examination of current adsorption models for Pb(II) and Cu(II)
  adsorption onto Fe3O4@Mg2Al-NO3 Layered Double Hydroxide in aqueous solution*.
  https://doi.org/10.1177/0263617417714166

The supplementary runner expects `data/zhao_marker_pixels.csv`,
`data/zhao_calibration.json`, and `data/zhao_frozen_queries.csv`.
These graphical readings are excluded along with the source image. Text S4
describes the calibration, reading rectangles, and monotone outer bounds.

## Numerical reference results

The empirical runners also load saved `results/` tables to check recalculated
values. These include source-framework predictions, attribution grids,
record-level comparisons, and the supplementary query registries. A source
workbook alone is therefore insufficient to run all verification commands.
The separately prepared numerical package is needed for those checks.
No downloader or complete empirical-input reconstruction is claimed here.

The code-only example and nine synthetic tests need none of these files.
Redistribution rights for the third-party empirical inputs and weights have
not been established, so they have not been uploaded.

