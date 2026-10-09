#!/usr/bin/env python3
"""Summarize the selected single-run Table 6 configs or an explicit seed cohort."""

import argparse
import csv
import json
import statistics
from pathlib import Path

try:
    from .run_table6_grid import BEST_SEEDS, SEEDS
except ImportError:
    from run_table6_grid import BEST_SEEDS, SEEDS


MODELS = {
    "RRNetHyperbolicLorentz": ("RResNet+PHDM", "Lorentz", "lorentz_fc"),
    "RRNetHyperbolic": ("RResNet+PHDM", "Poincare", "poincare_fc"),
    "RRNetSpherical": ("RResNet+PHDM", "Sphere", "sphere_exact"),
    "RRNetProjectedSpherical": ("RResNet+PHDM", "Projected sphere", "psphere_fc"),
    "RRNetGraphHyperbolicLorentz": ("RResNet Graph+PHDM", "Lorentz", "lorentz_fc"),
    "RRNetGraphHyperbolic": ("RResNet Graph+PHDM", "Poincare", "poincare_fc"),
    "RRNetGraphSpherical": ("RResNet Graph+PHDM", "Sphere", "sphere_exact"),
    "RRNetGraphProjectedSpherical": ("RResNet Graph+PHDM", "Projected sphere", "psphere_fc"),
}
METHODS = ("RResNet+PHDM", "RResNet Graph+PHDM")
GEOMETRIES = ("Lorentz", "Poincare", "Sphere", "Projected sphere")
GEOMETRY_KEYS = {"Lorentz": "lorentz", "Poincare": "poincare", "Sphere": "sphere", "Projected sphere": "projected_sphere"}


def load_rows(root, seeds, methods=METHODS):
    expected = {(m, g, s) for m in methods for g in GEOMETRIES
                for s in (seeds if seeds is not None else ((BEST_SEEDS[GEOMETRY_KEYS[g]],) if m == 'RResNet+PHDM' else SEEDS))}
    rows = {}
    for path in sorted(root.glob("**/runs/*.json")):
        result = json.loads(path.read_text())
        args = result.get("args", {})
        model = args.get("model")
        if result.get("status") != "complete" or model not in MODELS:
            continue
        method, geometry, projection = MODELS[model]
        if method not in methods or result.get('test_evaluations') != 1:
            continue
        if result.get('selection_policy') != 'validation_roc_ap_then_loss':
            continue
        if float(args.get("c", 0)) != 1.0 or args.get("proj_type") != projection:
            raise ValueError(f"Unexpected curvature or input map in {path}")
        seed = int(args["seed"])
        if (method, geometry, seed) not in expected:
            continue
        key = method, geometry, seed
        if key in rows:
            raise ValueError(f"Duplicate result for {key}: {path}")
        if int(result["best_epoch"]) < 1 or result.get("best_val_composite") is None:
            raise ValueError(f"Missing validation-selected checkpoint in {path}")
        rows[key] = {
            "method": method, "geometry": geometry, "seed": seed,
            "test_roc": 100 * float(result["test_at_best_val"]["roc"]),
            "test_ap": 100 * float(result["test_at_best_val"]["ap"]),
            "best_epoch": result["best_epoch"],
            "best_val_composite": result["best_val_composite"],
            "run_id": result["run_id"], "result_path": str(path),
        }
    missing = expected - rows.keys()
    if missing:
        raise ValueError(f"Incomplete Table 6 grid: {len(rows)}/{len(expected)}; missing {sorted(missing)}")
    return [rows[key] for key in sorted(expected)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", help="Summarize an explicit seed cohort instead of the selected single runs.")
    parser.add_argument("--architectures", default="rresnet")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",")) if args.seeds is not None else None
    if seeds is not None and (not seeds or len(set(seeds)) != len(seeds)):
        parser.error("Provide unique seeds")
    architectures = tuple(value.strip() for value in args.architectures.split(','))
    if not architectures or any(value not in ('rresnet', 'graph') for value in architectures):
        parser.error('Architectures must be rresnet and/or graph')
    methods = tuple('RResNet+PHDM' if value == 'rresnet' else 'RResNet Graph+PHDM' for value in architectures)
    rows = load_rows(args.run_root, seeds, methods)
    table = ["| Method | " + " | ".join(GEOMETRIES) + " |", "| --- | ---: | ---: | ---: | ---: |"]
    for method in methods:
        values = []
        for geometry in GEOMETRIES:
            scores = [r["test_roc"] for r in rows if (r["method"], r["geometry"]) == (method, geometry)]
            if len(scores) == 1:
                values.append(f"{scores[0]:.2f}")
            else:
                values.append(f"{statistics.mean(scores):.2f} +/- {statistics.stdev(scores):.2f}")
        table.append(f"| {method} | " + " | ".join(values) + " |")
    markdown = "\n".join(table) + "\n"
    print(markdown, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        with args.output.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
