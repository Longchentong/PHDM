#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.table3 import DATASETS, SETTINGS, TOTAL_EXAMPLES, Table3Setting


DEFAULT_SEEDS = ("paper",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate PHDM LoRA evaluation JSON outputs.")
    parser.add_argument("--run-root", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--models", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_seed(setting: Table3Setting, value: str) -> int:
    return setting.paper_seed if value == "paper" else int(value)


def result_path(experiment_dir: Path, setting: Table3Setting, dataset: str, seed: int) -> Path:
    run_name = setting.run_name(seed)
    return experiment_dir / (
        f"{setting.model_tag}-LoRA-{dataset}-32-{setting.lora_type}-"
        f"{setting.lora_alpha}-{run_name}.json"
    )


def read_accuracy(path: Path, expected: int) -> tuple[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < expected:
        raise ValueError(f"Incomplete result {path}: {len(data) if isinstance(data, list) else 'non-list'}")
    data = data[:expected]
    correct = sum(row.get("flag") is True or str(row.get("flag")).lower() == "true" for row in data)
    return correct, 100.0 * correct / expected


def load_run(experiment_dir: Path, setting: Table3Setting, seed: int) -> dict[str, float]:
    row: dict[str, float] = {}
    total_correct = 0
    for dataset, label, expected in DATASETS:
        correct, accuracy = read_accuracy(result_path(experiment_dir, setting, dataset, seed), expected)
        row[label] = accuracy
        total_correct += correct
    row["M.AVG"] = 100.0 * total_correct / TOTAL_EXAMPLES
    return row


def mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} +/- {statistics.stdev(values):.2f}"


def build_markdown(args: argparse.Namespace) -> str:
    experiment_dir = args.run_root / "experiment"
    labels = [label for _, label, _ in DATASETS] + ["M.AVG"]
    rows = []
    for key in args.models:
        setting = SETTINGS[key]
        run_rows = []
        for value in args.seeds:
            run_rows.append(load_run(experiment_dir, setting, resolve_seed(setting, value)))
        rows.append((setting.label, {label: mean_std([row[label] for row in run_rows]) for label in labels}))

    lines = [
        "| Base model | MAWPS (8.5%) | SVAMP (35.6%) | GSM8K (46.9%) | AQuA (9.0%) | M.AVG |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, summary in rows:
        lines.append(
            f"| {model} | {summary['MAWPS']} | {summary['SVAMP']} | {summary['GSM8K']} | "
            f"{summary['AQuA']} | **{summary['M.AVG']}** |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    markdown = build_markdown(args)
    print(markdown, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
