"""Run the six specified equilibrium controls without empirical input files."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import physics as p


def main():
    c0, dose, ce = 100.0, 2.0, 30.0
    bg_c0, bg_dose = np.meshgrid(
        [50.0, 100.0, 200.0], np.geomspace(0.1, 4.0, 201)
    )
    background = np.column_stack([bg_c0.ravel(), bg_dose.ravel()])
    rows = []
    for alpha in (-0.05, -0.02, 0.0, 0.02, 0.05, 0.1):
        baseline = p.solve(background, alpha)["q"].mean()
        dose_only = p.solve(
            np.column_stack([background[:, 0], np.full(len(background), dose)]), alpha
        )["q"].mean()
        c0_only = p.solve(
            np.column_stack([np.full(len(background), c0), background[:, 1]]), alpha
        )["q"].mean()
        fixed = p.solve(np.array([[c0, 1.0], [c0, dose]]), alpha)
        common = p.relation(ce, np.array([1.0, dose]), alpha)
        phi = 0.5 * (dose_only - baseline + fixed["q"][1] - c0_only)
        np.testing.assert_allclose(fixed["balance_error"], 0, atol=1e-12)
        rows.append((alpha, phi, 100 * (fixed["q"][1] / fixed["q"][0] - 1),
                     100 * (common[1] / common[0] - 1)))
    assert all(row[1] < 0 for row in rows)
    assert rows[0][3] > 0 and abs(rows[2][3]) < 1e-12 and rows[-1][3] < 0
    print("Synthetic controls; contributions in mg/g, changes in percent")
    print("alpha (L/g) | dose contribution | fixed C0 change | common Ce change")
    for row in rows:
        print(f"{row[0]:11.2f} | {row[1]:17.6f} | {row[2]:15.6f} | {row[3]:16.6f}")
    print("\nResidual-to-uptake relative error ratio R/(1-R):")
    for removal in (0.90, 0.95, 0.99):
        print(f"{100 * removal:.0f}% removal: {removal / (1 - removal):.0f}")


if __name__ == "__main__":
    main()

