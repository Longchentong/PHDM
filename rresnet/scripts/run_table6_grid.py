#!/usr/bin/env python3
"""Run the PHDM models in Table 6 at curvature magnitude one.

Defaults run the recorded single-seed RResNet configuration for each geometry.
Use --seeds to run an explicit cohort. Single-run scores do not establish the
paper's five-run means.
"""

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("rresnet", "graph")
GEOMETRIES = ("lorentz", "poincare", "sphere", "projected_sphere")
SEEDS = (1234, 1235, 1236, 1237, 1238)
BEST_SEEDS = {"lorentz": 1238, "poincare": 1234, "sphere": 1236, "projected_sphere": 1235}
REFERENCE_TRAINING = {"lr": 0.01, "eval_freq": 20, "patience": 100}
LORENTZ_TRAINING = {"lr": 0.0025, "eval_freq": 1, "patience": 2000}


def source_digest():
    result = hashlib.sha256()
    for directory in ('hgcn', 'rresnet'):
        for path in sorted((ROOT / directory).rglob('*.py')):
            result.update(str(path.relative_to(ROOT)).encode())
            result.update(path.read_bytes())
    return result.hexdigest()


def command_for(architecture, geometry, seed, args):
    defaults = LORENTZ_TRAINING if architecture == "rresnet" and geometry == "lorentz" else REFERENCE_TRAINING
    lr = args.lr if args.lr is not None else defaults["lr"]
    eval_freq = args.eval_freq if args.eval_freq is not None else defaults["eval_freq"]
    patience = args.patience if args.patience is not None else defaults["patience"]
    grad_clip = args.grad_clip
    settings = {
        "architecture": architecture,
        "geometry": geometry,
        "seed": seed,
        "curvature": 1.0,
        "epochs": args.epochs,
        "eval_freq": eval_freq,
        "patience": patience,
        "lr": lr,
        "weight_decay": args.weight_decay,
        "dim": 16,
        "hdim": 32,
        "num_layers": 2,
        "num_blocks": 3,
        "dropout": 0.2,
        "split_seed": 1234,
        "cuda": args.cuda,
        "source_sha256": getattr(args, 'source_sha256', None) or source_digest(),
        "selection_policy": "validation_roc_ap_then_loss",
        "evaluate_test": args.evaluate_test,
    }
    if grad_clip is not None:
        settings["grad_clip"] = grad_clip
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    run_id = f"{architecture}_{geometry}_seed{seed}_{digest}"
    metrics_path = args.output_root / "runs" / (run_id + ".json")
    checkpoint_path = args.output_root / "checkpoints" / (run_id + ".pt")
    command = [
        "bash", str(ROOT / "scripts" / "run_table6.sh"), architecture, geometry,
        "--dim", "16", "--hdim", "32", "--num-layers", "2", "--num-blocks", "3",
        "--act", "relu" if geometry in {"lorentz", "poincare"} else "None",
        "--bias", "1", "--dropout", "0.2", "--lr", str(lr),
        "--weight-decay", str(args.weight_decay), "--grad-clip", str(grad_clip),
        "--optimizer", "Adam", "--epochs", str(args.epochs),
        "--eval-freq", str(eval_freq), "--log-freq", "100",
        "--patience", str(patience), "--min-epochs", "100",
        "--lr-reduce-freq", str(args.epochs), "--gamma", "0.5", "--cuda", str(args.cuda),
        "--save", "0", "--seed", str(seed), "--split-seed", "1234",
        "--val-prop", "0.05", "--test-prop", "0.1", "--use-feats", "1",
        "--normalize-feats", "1", "--normalize-adj", "1", "--fail-on-nonfinite", "1",
        "--run-id", run_id, "--metrics-out", str(metrics_path),
        "--checkpoint-out", str(checkpoint_path), "--val-tie-break", "1",
        "--evaluate-test", str(args.evaluate_test),
    ]
    return run_id, metrics_path, command, settings


def completed(path, run_id, evaluate_test=1):
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (record.get("status") == "complete" and record.get("run_id") == run_id
            and record.get('selection_policy') == 'validation_roc_ap_then_loss'
            and record.get('test_evaluations') == evaluate_test
            and record.get('checkpoint') is not None
            and Path(record['checkpoint']).is_file()
            and hashlib.sha256(Path(record['checkpoint']).read_bytes()).hexdigest() == record.get('checkpoint_sha256'))


def run_one(job, args, gpu_slots):
    run_id, metrics_path, command, settings = job
    if completed(metrics_path, run_id, args.evaluate_test) and not args.force:
        return run_id, True, "skipped"
    gpu = gpu_slots.get()
    try:
        log_path = args.output_root / "logs" / (run_id + ".log")
        checkpoint_path = args.output_root / "checkpoints" / (run_id + ".pt")
        existing_paths = [path for path in (metrics_path, log_path, checkpoint_path) if path.exists()]
        if existing_paths:
            archive = args.output_root / 'interrupted' / f'{run_id}_{time.time_ns()}'
            archive.mkdir(parents=True)
            for path in existing_paths:
                shutil.move(str(path), str(archive / path.name))
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=gpu,
            PHDM_PYTHON=sys.executable,
            LOG_DIR=str(args.output_root / "model_logs"),
            PYTHONHASHSEED=str(settings["seed"]),
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        )
        with log_path.open("w") as handle:
            handle.write(json.dumps({"settings": settings, "command": command}, sort_keys=True) + "\n")
            handle.flush()
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        ok = result.returncode == 0 and completed(metrics_path, run_id, args.evaluate_test)
        return run_id, ok, "complete" if ok else f"failed rc={result.returncode} log={log_path}"
    finally:
        gpu_slots.put(gpu)


def parse_csv(value, allowed):
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values or any(value not in allowed for value in values):
        raise ValueError(f"Expected a nonempty selection from {allowed}, got {value}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", default="rresnet")
    parser.add_argument("--geometries", default=",".join(GEOMETRIES))
    parser.add_argument("--seeds", help="Override the selected single-run seeds with an explicit cohort.")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--evaluate-test", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--cuda", type=int, choices=(-1, 0), default=0)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "table6")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.workers_per_gpu) < 1 or any(
            value is not None and value < 1 for value in (args.eval_freq, args.patience)):
        parser.error("Epoch, evaluation, patience and worker counts must be positive")
    args.output_root = args.output_root.resolve()
    args.source_sha256 = source_digest()
    architectures = parse_csv(args.architectures, ARCHITECTURES)
    geometries = parse_csv(args.geometries, GEOMETRIES)
    seeds = tuple(int(value) for value in args.seeds.split(",")) if args.seeds is not None else None
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus or (seeds is not None and (not seeds or len(set(seeds)) != len(seeds))):
        parser.error("GPU and seed lists must be nonempty, and seeds must be unique")
    jobs = [command_for(a, g, s, args) for a in architectures for g in geometries
            for s in (seeds if seeds is not None else ((BEST_SEEDS[g],) if a == 'rresnet' else SEEDS))]
    if args.dry_run:
        import shlex
        for _, _, command, _ in jobs:
            print(shlex.join(command))
        print(f"{len(jobs)} jobs")
        return 0
    for name in ("runs", "logs", "model_logs"):
        (args.output_root / name).mkdir(parents=True, exist_ok=True)
    gpu_slots = queue.Queue()
    for gpu in gpus:
        for _ in range(args.workers_per_gpu):
            gpu_slots.put(gpu)
    manifest = [{"run_id": j[0], "metrics": str(j[1]), "settings": j[3]} for j in jobs]
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    failures = []
    with ThreadPoolExecutor(max_workers=len(gpus) * args.workers_per_gpu) as executor:
        futures = [executor.submit(run_one, job, args, gpu_slots) for job in jobs]
        for future in as_completed(futures):
            run_id, ok, status = future.result()
            print(f"[{status}] {run_id}", flush=True)
            if not ok:
                failures.append(run_id)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
