#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

try:
    from .paper import DATASETS, METHODS, PROTOCOLS, ROOT, VALIDATION_POLICIES
except ImportError:
    from paper import DATASETS, METHODS, PROTOCOLS, ROOT, VALIDATION_POLICIES


DISPLAY = {"phdm": "Hypformer + PHDM"}


def load_metrics(run_root, protocol):
    records = {}
    for path in sorted(run_root.glob("*/metrics.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            record.get("status") != "completed"
            or record.get("selection_policy") not in VALIDATION_POLICIES
            or record.get("protocol") != protocol
            or record.get("method") not in METHODS
        ):
            continue
        records[(record["method"], record["dataset"])] = record
    return records


def metric_cell(record):
    if record is None:
        return "-"
    if record.get('runs') == 1:
        return f"{record['score']:.2f}"
    return f"{record['val_best_test_mean']:.2f} +/- {record['val_best_test_std']:.2f}"


def build_report(records, protocol):
    lines = [
        "# Hypformer node-classification reproduction",
        "",
        f"Protocol: `{protocol}`. Completed cells: {len(records)} / {len(METHODS) * len(DATASETS)}.",
        "Checkpoint selection: validation accuracy, with optional validation-loss tie handling; test evaluated once after reload.",
        "",
        "| Model | Airport | Cora | Citeseer | PubMed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        cells = [metric_cell(records.get((method, dataset))) for dataset in DATASETS]
        lines.append(f"| {DISPLAY[method]} | " + " | ".join(cells) + " |")
    missing = [
        f"{method}/{dataset}"
        for method in METHODS
        for dataset in DATASETS
        if (method, dataset) not in records
    ]
    if missing:
        lines.extend(["", "## Missing cells", "", *[f"- {cell}" for cell in missing]])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize Hypformer runs as Markdown.")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="paper")
    parser.add_argument("--run-root", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "report.md")
    args = parser.parse_args()
    records = load_metrics(args.run_root, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(records, args.protocol))
    print(args.output)


if __name__ == "__main__":
    main()
