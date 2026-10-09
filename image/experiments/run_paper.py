#!/usr/bin/env python3
import argparse
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .paper import CODE_DIR, DATASETS, GEOMETRIES, METHODS, ROOT, build_jobs
except ImportError:
    from paper import CODE_DIR, DATASETS, GEOMETRIES, METHODS, ROOT, build_jobs


RESULT_RE = re.compile(
    r"Results: Loss=([0-9.eE+-]+), Acc@1=([0-9.eE+-]+), Acc@5=([0-9.eE+-]+)"
)
BEST_RE = re.compile(r"Best epoch = ([0-9]+), with Acc@1=([0-9.eE+-]+)")


def parse_csv(value, allowed, cast=str):
    values = tuple(cast(part.strip()) for part in value.split(",") if part.strip())
    invalid = sorted(set(values) - set(allowed)) if allowed is not None else []
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported values: {', '.join(map(str, invalid))}")
    return values


def parse_training_log(text):
    best_match = BEST_RE.findall(text)
    result_matches = RESULT_RE.findall(text)
    if not best_match:
        raise ValueError("Training log does not contain a best-epoch metric")
    best_epoch, best_validation_acc1 = best_match[-1]
    metrics = {
        "best_epoch": int(best_epoch),
        "best_validation_acc1": float(best_validation_acc1),
    }
    if result_matches:
        loss, acc1, acc5 = result_matches[-1]
        metrics.update(
            best_test_loss=float(loss),
            best_test_acc1=float(acc1),
            best_test_acc5=float(acc5),
        )
    # HCNN uses the test loader as val_loader when no validation split is
    # requested, so the best-epoch value is the paper-compatible metric.
    metrics["accuracy"] = metrics["best_validation_acc1"]
    return metrics


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def command_for(job, python, epochs, output_dir):
    return [
        python,
        str(CODE_DIR / "classification" / "train.py"),
        "-c",
        str(job.config),
        "--exp_name",
        job.run_id,
        "--dataset",
        job.dataset,
        "--seed",
        str(job.seed),
        "--clip_features",
        str(job.clip_features),
        "--num_epochs",
        str(epochs),
        "--device",
        "cuda:0",
        "--output_dir",
        str(output_dir),
    ]


def remove_checkpoints(run_dir):
    for checkpoint in run_dir.glob("*.pth"):
        checkpoint.unlink()


def run_job(job, gpu_slots, args, manifest_lock):
    run_dir = args.output_root / job.run_id
    metrics_path = run_dir / "metrics.json"
    log_path = run_dir / "train.log"
    if metrics_path.exists() and not args.force:
        try:
            existing = json.loads(metrics_path.read_text())
            if existing.get("status") == "completed":
                return job.run_id, "skipped", existing.get("accuracy")
        except (OSError, json.JSONDecodeError):
            pass

    run_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpu_slots.get()
    command = command_for(job, args.python, args.epochs, run_dir)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        HYPERCV_EUCLIDEAN_TO_LORENTZ=job.mapping,
        HYPERCV_EUCLIDEAN_TO_POINCARE=job.mapping,
        HYPERCV_PHDM_INPUT_SCALE=str(args.phdm_input_scale),
        PYTHONUNBUFFERED="1",
    )
    started = time.time()
    running = {
        "status": "running",
        "run_id": job.run_id,
        "method": job.method,
        "geometry": job.geometry,
        "dataset": job.dataset,
        "seed": job.seed,
        "mapping": job.mapping,
        "clip_features": job.clip_features,
        "phdm_input_scale": args.phdm_input_scale,
        "gpu": str(gpu),
        "epochs": args.epochs,
        "command": command,
        "started_at": started,
    }
    atomic_json(metrics_path, running)

    try:
        with log_path.open("w") as log_file:
            process = subprocess.run(
                command,
                cwd=CODE_DIR,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.time() - started
        payload = dict(running, returncode=process.returncode, elapsed_seconds=elapsed)
        if process.returncode == 0:
            payload.update(parse_training_log(log_path.read_text()), status="completed")
            if not args.keep_checkpoints:
                remove_checkpoints(run_dir)
        else:
            payload["status"] = "failed"
        atomic_json(metrics_path, payload)
        with manifest_lock:
            with (args.output_root / "manifest.jsonl").open("a") as manifest:
                manifest.write(json.dumps(payload, sort_keys=True) + "\n")
        return job.run_id, payload["status"], payload.get("accuracy")
    except Exception as error:
        payload = dict(
            running,
            status="failed",
            elapsed_seconds=time.time() - started,
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(metrics_path, payload)
        return job.run_id, "failed", None
    finally:
        gpu_slots.put(gpu)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Run one historical best-seed PHDM configuration per image-classification cell."
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--geometries", default=",".join(GEOMETRIES))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", help="Override the selected seed for every requested cell.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--phdm-input-scale", type=float, default=0.1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = get_arguments()
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be positive")
    gpus = tuple(part.strip() for part in args.gpus.split(",") if part.strip())
    if not gpus:
        raise SystemExit("At least one GPU is required")
    methods = parse_csv(args.methods, METHODS)
    geometries = parse_csv(args.geometries, GEOMETRIES)
    datasets = parse_csv(args.datasets, DATASETS)
    seeds = parse_csv(args.seeds, None, int) if args.seeds is not None else None
    if seeds is not None and (not seeds or any(seed < 0 for seed in seeds)):
        raise SystemExit("Seeds must be non-negative integers")
    jobs = build_jobs(methods, geometries, datasets, seeds)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    args.output_root = args.output_root.resolve()

    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            command = command_for(job, args.python, args.epochs, args.output_root / job.run_id)
            print(f"GPU {gpu}: {shlex.join(command)}")
        print(f"{len(jobs)} jobs")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    gpu_slots = queue.Queue()
    for gpu in gpus:
        for _ in range(args.workers_per_gpu):
            gpu_slots.put(gpu)
    manifest_lock = threading.Lock()
    failures = []
    max_workers = len(gpus) * args.workers_per_gpu
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_job, job, gpu_slots, args, manifest_lock) for job in jobs
        ]
        for future in as_completed(futures):
            run_id, status, accuracy = future.result()
            metric = "" if accuracy is None else f" accuracy={accuracy:.2f}"
            print(f"[{status}] {run_id}{metric}", flush=True)
            if status == "failed":
                failures.append(run_id)
    if failures:
        raise SystemExit(f"{len(failures)} runs failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
