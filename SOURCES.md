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
They are excluded from this release. The preparation command below reconstructs
them from the pinned source files; Supporting Information Texts S1 and S2 describe
their use. `models.py` is an
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

## Preparing model inputs from the original files

Obtain the files at the revisions above, preserving their repository paths.
For biochar, the required files are `scripts/HMI_data.xlsx`, `scripts/utils.py`,
`scripts/interpretation.py`, `scripts/results/figures/kernel_shap_ftt.csv`, and
`config.json` and `weights/weights_110_0.08815.hdf5` within
`scripts/results/ft_transformer_selected_inputs_log/`.
For activated carbon, obtain `Dataset. S.Lamsiah.xlsx` and
`HeavyMetalAdsorptionPredictor_S_Lamsiah.ipynb` at the repository root.
The script checks each file against the inspected SHA-256 fingerprint before
reading it. It does not execute the source Python or notebook.

```bash
python src/prepare_inputs.py \
  --biochar-source /path/to/envai101 \
  --activated-carbon-source /path/to/Heavy-Metal-Adsorption-Activated-Carbon-ML \
  --output /path/to/new/prepared-data
```

The output directory must not already exist. Both workbooks use sheet index 0;
no records are filtered or reordered in the prepared tables. Biochar column
names and selected features are extracted from the verified source loader and
interpretation script. Numerical fields are converted to floats, categorical
fields to strings, and category order comes from the saved model configuration.
The workbook already uses grams and liters for loading and volume; no additional
unit conversion is applied. Activated-carbon columns retain their original names
and units, including the newline in the total-pore-volume header.

The biochar split is the seed-1000 permutation, with the first 70% training.
Activated carbon uses the seed-42 stratified 70/15/15 split with 29 response
quantiles; preprocessing is fitted later on training data only. A split registry
records both zero-based input indices and Excel row numbers (index + 2).
`preparation_report.json` contains column mappings, schemas, source fingerprints,
and checksums of the prepared files. The model settings distinguish reuse of
the biochar checkpoint from refitting activated carbon. This command prepares
the two model datasets, not the separately digitized Zhao inputs.

## Optional numerical reference results

The empirical runners can compare their outputs with saved reference results
when `--reference-dir` is supplied. Without that option they calculate from
inputs only. Saved predictions and result tables do not supply numerical inputs
to the primary, sensitivity, or supplementary computations. The graph-reading
coordinates, calibration, and frozen query registry remain required inputs,
distinct from reference outputs. There is no downloader.

Biochar predictions are recomputed from unchanged weights using NumPy. This
replay can differ slightly from the original TensorFlow implementation through
floating-point arithmetic; it is not a new model fit. Archived source SHAP is
retained for the historical-explanation comparison. Independently calculated
two-group contributions and sampled near-zero checks are separate analyses.

The code-only example and nine synthetic tests need none of these files.
Redistribution rights for the third-party empirical inputs and weights have
not been established, so they have not been uploaded.
