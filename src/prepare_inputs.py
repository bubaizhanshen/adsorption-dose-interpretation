"""Prepare empirical inputs from pinned author files, without running their code.

Source workbooks and weights must be acquired separately. This script does not
download or redistribute them. All column mappings are read from verified source
files; prepared files are written only to a new directory.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

os.umask(0o077)
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


BIOCHAR_COMMIT = "fc6fd3a9b2adcb5c05a733222201c3707b41d907"
CARBON_COMMIT = "99d7d12a6c60aa9e86f68027c440cd2587d2c9e5"
BIOCHAR_FILES = {
    "scripts/HMI_data.xlsx": "2074a592c184aa680f321d5d28f4c8371dba1c249c915a67352cc7e2e4fc725d",
    "scripts/utils.py": "43aff01ae2924bf894f03342e2d3e2e7013115f15eee1cbe8f7d721ab28d775e",
    "scripts/interpretation.py": "592479d2ac1f508b9457a5e05976e673f53171a557f870817bde2573f87d51b4",
    "scripts/results/ft_transformer_selected_inputs_log/config.json": "de89789f24fe966e6078055edf4a5a15667ffcce545151fb724fb6a96af892d7",
    "scripts/results/ft_transformer_selected_inputs_log/weights/weights_110_0.08815.hdf5": "0e17d49e3c5589417a797cb3ae736811e27b559a5c6cb8963cb4c8ab8485dd1e",
    "scripts/results/figures/kernel_shap_ftt.csv": "298bc81ed6116d0027777430233604cc1cb3c7a900900095ecfeabe34c24fde7",
}
CARBON_FILES = {
    "Dataset. S.Lamsiah.xlsx": "fd5c848faf2158115dc276fb15500a6cea244980881f9cf880bb383a1fa82581",
    "HeavyMetalAdsorptionPredictor_S_Lamsiah.ipynb": "dd338b6a0ab4009907178295691d8c75211a3ce23329d0319d89a69d64f17ad2",
}
FEATURES = [
    "Dose (g/L)", "Initial concentration (mg/L)", "Contact time (min)", "pH",
    "BET surface area (m\u00b2/g)", "Pore diameter (nm)", "Total pore volume (cm\u00b3/g)\n",
    "Activation temperature (\u00b0C)", "van der Waals radius (nm)", "Temperature (\u00b0C)",
    "Molar mass (g/mol)", "Hydrated radius (nm)", "Electronegativity (Pauling)",
]
PARAMETERS = dict(n_estimators=150, learning_rate=.10, max_depth=15,
                  subsample=.7, max_features=.7, min_samples_split=10,
                  min_samples_leaf=6, loss="squared_error", random_state=42,
                  validation_fraction=.15, n_iter_no_change=20, tol=.001)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(root, expected):
    for filename, checksum in expected.items():
        path = root / filename
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Missing or changed pinned source: {path}")


def literal_assignment(nodes, name):
    matches = [n for n in nodes if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(matches) != 1:
        raise ValueError(f"Expected one literal assignment for {name}")
    return ast.literal_eval(matches[0].value)


def prepare(biochar, carbon, output):
    verify_sources(biochar, BIOCHAR_FILES)
    verify_sources(carbon, CARBON_FILES)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")

    source = ast.parse((biochar / "scripts/utils.py").read_text())
    loader = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_load_data")
    columns = next(ast.literal_eval(n.value) for n in ast.walk(loader)
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                           and t.value.id == "data" and t.attr == "columns" for t in n.targets))
    nodes = ast.parse((biochar / "scripts/interpretation.py").read_text()).body
    selected = literal_assignment(nodes, "selected_features")
    numeric = [x for x in literal_assignment(nodes, "NUMERIC_FEATURES") if x in selected]
    categories = [x for x in literal_assignment(nodes, "CAT_FEATURES") if x in selected]
    raw = pd.read_excel(biochar / "scripts/HMI_data.xlsx", sheet_name=0)
    column_mapping = dict(zip(map(str, raw.columns), columns))
    if raw.shape != (1518, len(columns)):
        raise ValueError("Unexpected biochar workbook shape")
    raw.columns = columns
    raw[numeric] = raw[numeric].astype(float)
    raw[categories] = raw[categories].astype(str)
    raw["qe"] = raw.qe.astype(float)
    if not np.isfinite(raw[numeric + ["qe"]].to_numpy()).all() or (raw.qe <= 0).any():
        raise ValueError("Invalid biochar predictors or log-response inputs")
    order = np.random.RandomState(1000).permutation(len(raw))
    cut = int(len(raw) * .7)
    train, test = order[:cut], order[cut:]
    config = json.loads((biochar / "scripts/results/ft_transformer_selected_inputs_log/config.json").read_text())
    vocabulary = config["config"]["model"]["layers"]["FTTransformer"]["config"]["cat_vocabulary"]
    for column in categories:
        if not set(raw[column]).issubset(vocabulary[column]):
            raise ValueError(f"Unmapped category in {column}")
    weights = {}
    weight_path = biochar / "scripts/results/ft_transformer_selected_inputs_log/weights/weights_110_0.08815.hdf5"
    with h5py.File(weight_path, "r") as handle:
        def collect(name, node):
            if isinstance(node, h5py.Dataset):
                weights[name] = node[...]
        handle.visititems(collect)
    if sum(x.size for x in weights.values()) != 37739:
        raise ValueError("Unexpected transformer parameter count")
    saved = pd.read_csv(biochar / "scripts/results/figures/kernel_shap_ftt.csv", index_col=0)
    if len(saved) != len(test):
        raise ValueError("Saved SHAP row count does not match the source test split")

    ac = pd.read_excel(carbon / "Dataset. S.Lamsiah.xlsx", sheet_name=0)
    x = ac[FEATURES].to_numpy(float)
    y = ac["Adsorption capacity (mg/g)"].to_numpy(float)
    if len(ac) != 1528 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid activated-carbon inputs")
    if np.any(x[:, :2] <= 0):
        raise ValueError("Dose and initial concentration must be positive")
    bins = pd.qcut(y, q=29, labels=False, duplicates="drop")
    ac_train, temporary = train_test_split(np.arange(len(ac)), test_size=.30,
                                          random_state=42, stratify=bins)
    validation, ac_test = train_test_split(temporary, test_size=.50,
                                          random_state=42, stratify=bins[temporary])
    settings = {
        "biochar": {"numeric_features": numeric, "categorical_features": categories,
                    "vocabulary": vocabulary, "split_seed": 1000, "test_fraction": .3,
                    "training_records": len(train), "test_records": len(test),
                    "original_checkpoint_replayed": True},
        "activated_carbon": {"features": FEATURES, "parameters": PARAMETERS,
                             "split_seed": 42, "stratification_quantiles": 29,
                             "training_records": len(ac_train), "validation_records": len(validation),
                             "test_records": len(ac_test), "reported_settings_refitted": True},
        "units": {"uptake": "mg/g", "dose": "g/L", "initial_and_residual_concentration": "mg/L",
                  "biochar_loading": "g", "biochar_solution_volume": "L"},
        "release_status": "local only; no upload performed",
    }
    output.mkdir(mode=0o700, parents=True)
    raw.to_csv(output / "biochar_records.csv", index=False)
    ac.to_csv(output / "activated_carbon_records.csv", index=False)
    saved.to_csv(output / "biochar_saved_shap.csv", index=False)
    np.savez_compressed(output / "biochar_weights.npz", **weights)
    (output / "model_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    splits = []
    for case, partitions in (("biochar", {"train": train, "test": test}),
                             ("activated_carbon", {"train": ac_train, "validation": validation, "test": ac_test})):
        for name, indices in partitions.items():
            splits.extend({"case": case, "partition": name, "position": j,
                           "source_index": int(i), "source_excel_row": int(i + 2)}
                          for j, i in enumerate(indices))
    pd.DataFrame(splits).to_csv(output / "split_registry.csv", index=False)
    report = {"source_commits": {"biochar": BIOCHAR_COMMIT, "activated_carbon": CARBON_COMMIT},
              "source_sha256": {"biochar": BIOCHAR_FILES, "activated_carbon": CARBON_FILES},
              "sheet_index": 0, "biochar_column_mapping": column_mapping,
              "biochar_schema": {c: str(t) for c, t in raw.dtypes.items()},
              "activated_carbon_schema": {c: str(t) for c, t in ac.dtypes.items()},
              "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biochar-source", type=Path, required=True)
    parser.add_argument("--activated-carbon-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.biochar_source.resolve(), args.activated_carbon_source.resolve(), args.output.resolve())
    print(json.dumps(report["output_sha256"], indent=2))
