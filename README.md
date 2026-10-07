# Adsorbent-dose interpretation

Code accompanying **Mass Balance Limits Mechanistic Inference from
Adsorbent-Dose Attributions**.

The study asks whether a negative dose attribution identifies a change in the
adsorption response, and whether accurate uptake predictions also resolve the
contaminant remaining in solution. The code compares uptake and removal
explanations, evaluates specified equilibrium responses, and calculates
residual-concentration errors.

## Quick start: no external data needed

Tested with Python 3.13. Install the numerical dependencies, then run the
synthetic example and tests from this directory:

```bash
python -m pip install -r requirements.txt
python examples/equilibrium_controls.py
python -m unittest discover -s tests -v
```

The example prints six exact dose contributions, uptake changes at fixed
initial concentration and common equilibrium concentration, and the relative
error amplification implied by mass balance. All example values are generated
from specified functions; they are not measurements. No network access is used.

Nine tests run on synthetic inputs. Four further tests check manuscript record
registries and are explicitly skipped when the separate inputs are absent.
Those four checks are not counted as passed in a code-only checkout.

## Files

```text
examples/equilibrium_controls.py  Runnable exact-function comparison
src/physics.py                  Mass balance, attribution, matching, root solving
src/auxiliary_physics.py         Kinetic, competition, graphical-bound calculations
src/models.py                   Independent NumPy transformer inference
src/reproduce.py                Primary manuscript calculations
src/review_audit.py              Background and numerical sensitivity checks
src/reproduce_supplement.py      Supplementary controls and digitization checks
src/plot_figures.py              Figure generation from numerical results
tests/test_physics.py            Synthetic and optional record-level tests
SOURCES.md                      Third-party sources and reproduction requirements
requirements.txt               Verified numerical environment
```

## Manuscript calculations

The `src/` analysis files are the same as those used in the local numerical
verification. The full manuscript calculations also require the prepared
`data/` and numerical reference `results/` directories described in
[SOURCES.md](SOURCES.md). After placing those directories beside `src/`:

```bash
python src/reproduce.py --output reproduced
python src/review_audit.py
python src/reproduce_supplement.py
python src/plot_figures.py
python -m unittest discover -s tests -v
```

The primary runner refuses to overwrite a nonempty output directory. The
sensitivity runner writes `results/review_audit/`; use a working copy of the
inputs when running it. Figure generation reads saved numerical results, so
plotting alone is not a numerical reproduction.

## Release scope

This repository contains code, instructions, and synthetic examples only.
It excludes manuscripts, article PDFs, experimental tables, digitized readings,
saved source explanations, trained weights, and manuscript result tables.
Full empirical reproduction consequently requires inputs obtained separately;
the quick-start example reproduces the specified physical controls only.

Source data and models retain their original authorship and terms. Public
availability of a source is not treated as permission to redistribute it.

