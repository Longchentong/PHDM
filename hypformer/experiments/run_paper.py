#!/usr/bin/env python3
import argparse
import json
import math
import os
import queue
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .paper import DATASETS, METHODS, PROTOCOLS, ROOT, SOURCE_DIR, VALIDATION_POLICIES, build_jobs
except ImportError:
    from paper import DATASETS, METHODS, PROTOCOLS, ROOT, SOURCE_DIR, VALIDATION_POLICIES, build_jobs


SUMMARY_RE = re.compile(
    r"Highest Test:\s*([0-9.]+)\s*(?:\+/-|\u00b1)\s*([0-9.]+).*?"
    r"Final Test:\s*([0-9.]+)\s*(?:\+/-|\u00b1)\s*([0-9.]+)"
)
RUN_HIGHEST_RE = re.compile(r"Run\s+\d+:.*?Highest Test:\s*([0-9.]+)")
RUN_FINAL_RE = re.compile(r"Final Train:\s*[0-9.]+\s+Final Test:\s*([0-9.]+)")


def parse_csv(value, allowed):
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported values: {', '.join(invalid)}")
    return values


def parse_training_log(text):
    records = [json.loads(line.split(': ', 1)[1]) for line in text.splitlines()
               if line.startswith('PHDM_RUN_RESULT: ')]
    if records:
        policies = {record.get('selection_policy') for record in records}
        if len(policies) != 1 or not policies.issubset(VALIDATION_POLICIES) or any(
               record.get('test_evaluations') != 1
               or not math.isfinite(record['test_accuracy']) for record in records):
            raise ValueError('Invalid validation-selected run records')
        scores = [record['test_accuracy'] for record in records]
        return {
            'selection_policy': records[0]['selection_policy'],
            'val_best_test_mean': statistics.mean(scores),
            'val_best_test_std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'score': statistics.mean(scores),
            'run_results': records,
        }
    summaries = SUMMARY_RE.findall(text)
    if summaries:
        highest_mean, highest_std, final_mean, final_std = summaries[-1]
        return {
            "highest_test_mean": float(highest_mean),
            "highest_test_std": float(highest_std),
            "final_test_mean": float(final_mean),
            "final_test_std": float(final_std),
            "score": float(final_mean),
            "selection_policy": "historical_validation_accuracy",
        }
    highest = RUN_HIGHEST_RE.findall(text)
    final = RUN_FINAL_RE.findall(text)
    if highest and final:
        return {
            "highest_test_mean": float(highest[-1]),
            "highest_test_std": 0.0,
            "final_test_mean": float(final[-1]),
            "final_test_std": 0.0,
            "score": float(final[-1]),
            "selection_policy": "historical_validation_accuracy",
        }
    raise ValueError("Training log does not contain Hypformer result metrics")


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def experiment_id(job, args):
    suffix = f"run{job.selected_run}" if args.runs is None else f"runs{args.runs}"
    return f"{job.run_id}_{suffix}"


def command_for(job, args, run_dir):
    config = job.config
    if args.epochs is not None:
        config["epochs"] = args.epochs
    config.update(
        dataset=job.dataset,
        method="hypformer",
        input_map=job.input_map,
        runs=job.selected_run if args.runs is None else args.runs,
        selected_run=job.selected_run if args.runs is None else 0,
        device=0,
        data_dir=str(args.data_dir),
        run_id=experiment_id(job, args),
        save_result=0,
        checkpoint_dir=str(run_dir / 'checkpoints'),
        result_file=str(run_dir / 'selected_runs.json'),
    )
    command = [args.python, str(SOURCE_DIR / "main.py")]
    for key, value in config.items():
        command.extend((f"--{key}", str(value)))
    return command


def run_job(job, gpu_slots, args, manifest_lock):
    run_id = experiment_id(job, args)
    run_dir = args.output_root / run_id
    reported_runs = 1 if args.runs is None else args.runs
    metrics_path = run_dir / "metrics.json"
    log_path = run_dir / "train.log"
    command = command_for(job, args, run_dir)
    expected_policy = ('validation_accuracy_then_loss' if job.config.get('val_tie_break')
                       else 'validation_accuracy')
    if metrics_path.exists() and not args.force:
        try:
            existing = json.loads(metrics_path.read_text())
            if (existing.get("status") == "completed"
                    and existing.get('selection_policy') == expected_policy
                    and existing.get('runs') == reported_runs
                    and existing.get('command') == command):
                return run_id, "skipped", existing.get("score")
        except (OSError, json.JSONDecodeError):
            pass

    run_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpu_slots.get()
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        HYPFORMER_INPUT_MAP=job.input_map,
        PYTHONHASHSEED=str(job.config["seed"]),
        PYTHONUNBUFFERED="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    started = time.time()
    payload = {
        "status": "running",
        "run_id": run_id,
        "protocol": job.protocol,
        "method": job.method,
        "dataset": job.dataset,
        "input_map": job.input_map,
        "runs": reported_runs,
        "training_runs": job.selected_run if args.runs is None else args.runs,
        "selected_run": job.selected_run if args.runs is None else None,
        "gpu": str(gpu),
        "config": job.config,
        "command": command,
        "started_at": started,
    }
    atomic_json(metrics_path, payload)

    try:
        with log_path.open("w") as log_file:
            process = subprocess.run(
                command,
                cwd=run_dir,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        payload.update(
            returncode=process.returncode,
            elapsed_seconds=time.time() - started,
        )
        if process.returncode == 0:
            metrics = parse_training_log(log_path.read_text())
            if (metrics.get('selection_policy') != expected_policy
                    or len(metrics.get('run_results', [])) != reported_runs):
                raise ValueError('Incomplete validation-selected experiment')
            payload.update(metrics, status="completed")
        else:
            payload["status"] = "failed"
        atomic_json(metrics_path, payload)
        with manifest_lock:
            with (args.output_root / "manifest.jsonl").open("a") as manifest:
                manifest.write(json.dumps(payload, sort_keys=True) + "\n")
        return run_id, payload["status"], payload.get("score")
    except Exception as error:
        payload.update(
            status="failed",
            elapsed_seconds=time.time() - started,
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(metrics_path, payload)
        return run_id, "failed", None
    finally:
        gpu_slots.put(gpu)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Run one selected PHDM configuration per Table 5 dataset."
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--protocol", choices=PROTOCOLS, default="paper")
    parser.add_argument("--runs", type=int,
                        help="Override single-run replay with this many continuous training/test runs.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = get_arguments()
    if args.workers_per_gpu < 1 or (args.runs is not None and args.runs < 1):
        raise SystemExit("Worker and run counts must be positive")
    gpus = tuple(part.strip() for part in args.gpus.split(",") if part.strip())
    if not gpus:
        raise SystemExit("At least one GPU is required")
    methods = parse_csv(args.methods, METHODS)
    datasets = parse_csv(args.datasets, DATASETS)
    jobs = build_jobs(methods, datasets, args.protocol)
    args.data_dir = args.data_dir.resolve()
    args.output_root = args.output_root.resolve()

    airport_file = args.data_dir / "hgcn_data" / "airport" / "airport.p"
    if "airport" in datasets and not args.dry_run and not airport_file.exists():
        raise SystemExit(
            f"Airport data is missing at {airport_file}. Run scripts/download_airport.py first."
        )

    if args.dry_run:
        for index, job in enumerate(jobs):
            gpu = gpus[index % len(gpus)]
            run_dir = args.output_root / experiment_id(job, args)
            print(f"GPU {gpu}: {shlex.join(command_for(job, args, run_dir))}")
        if args.runs is None:
            print(f"{len(jobs)} selected test runs; preceding training runs replay the original RNG sequence")
        else:
            print(f"{len(jobs)} jobs x {args.runs} internal runs")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    gpu_slots = queue.Queue()
    for gpu in gpus:
        for _ in range(args.workers_per_gpu):
            gpu_slots.put(gpu)
    manifest_lock = threading.Lock()
    failures = []
    with ThreadPoolExecutor(max_workers=len(gpus) * args.workers_per_gpu) as executor:
        futures = [
            executor.submit(run_job, job, gpu_slots, args, manifest_lock) for job in jobs
        ]
        for future in as_completed(futures):
            run_id, status, score = future.result()
            metric = "" if score is None else f" score={score:.2f}"
            print(f"[{status}] {run_id}{metric}", flush=True)
            if status == "failed":
                failures.append(run_id)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
