#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from .paper import BEST_SEEDS, DATASETS, GEOMETRIES, METHODS, ROOT
except ImportError:
    from paper import BEST_SEEDS, DATASETS, GEOMETRIES, METHODS, ROOT


DISPLAY = {
    ("phdm", "poincare"): "Hybrid Poincare + PHDM",
    ("phdm", "lorentz"): "Hybrid Lorentz + PHDM",
}


def load_metrics(run_root):
    records = []
    for path in sorted(run_root.glob("*/metrics.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            record.get("status") == "completed"
            and record.get("method") in METHODS
            and "accuracy" in record
            and record.get("seed") == BEST_SEEDS.get((record.get("geometry"), record.get("dataset")))
        ):
            records.append(record)
    return records


def format_stat(values):
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.2f} +/- {deviation:.2f}"


def build_report(records):
    grouped = defaultdict(list)
    for record in records:
        key = (record["method"], record["geometry"], record["dataset"])
        grouped[key].append(float(record["accuracy"]))

    lines = [
        "# Image-classification reproduction",
        "",
        f"Completed single-seed runs: {len(records)} / {len(METHODS) * len(GEOMETRIES) * len(DATASETS)}.",
        "",
        "| Model | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |",
        "| --- | ---: | ---: | ---: |",
    ]
    for geometry in GEOMETRIES:
        for method in METHODS:
            cells = [format_stat(grouped[(method, geometry, dataset)]) for dataset in DATASETS]
            lines.append(f"| {DISPLAY[(method, geometry)]} | " + " | ".join(cells) + " |")

    missing = []
    for geometry in GEOMETRIES:
        for method in METHODS:
            for dataset in DATASETS:
                count = len(grouped[(method, geometry, dataset)])
                if count != 1:
                    missing.append(f"{method}/{geometry}/{dataset}: {count}/1")
    if missing:
        lines.extend(["", "## Incomplete cells", "", *[f"- {item}" for item in missing]])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize completed image runs as Markdown.")
    parser.add_argument("--run-root", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "report.md")
    args = parser.parse_args()
    records = load_metrics(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(records))
    print(args.output)


if __name__ == "__main__":
    main()
