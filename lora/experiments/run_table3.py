#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.table3 import DATASETS, SETTINGS, TARGET_MODULES, Table3Setting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PHDM LoRA model/seed.")
    parser.add_argument("--model", choices=SETTINGS, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--phase", choices=("train", "eval", "both"), default="both")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    parser.add_argument("--run-root", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--model-parallel-max-memory", default="58GiB")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def bool_text(value: bool) -> str:
    return "True" if value else "False"


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_command(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(f"$ {command_text(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def adapter_exists(output_dir: Path) -> bool:
    return any((output_dir / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))


def training_command(
    setting: Table3Setting,
    base_model: str,
    seed: int,
    output_dir: Path,
    gpus: list[str],
    max_memory: str,
) -> list[str]:
    common = [
        "--base_model",
        base_model,
        "--data_path",
        str(ROOT / setting.data_file),
        "--output_dir",
        str(output_dir),
        "--batch_size",
        "16",
        "--micro_batch_size",
        str(setting.micro_batch_size),
        "--num_epochs",
        str(setting.epochs),
        "--learning_rate",
        str(setting.learning_rate),
        "--cutoff_len",
        "256",
        "--val_set_size",
        "120",
        "--seed",
        str(seed),
        "--use_gradient_checkpointing",
        bool_text(setting.gradient_checkpointing),
        "--adapter_name",
        "lora",
        "--lora_r",
        "32",
        "--lora_alpha",
        str(setting.lora_alpha),
        "--lora_dropout",
        str(setting.lora_dropout),
        "--target_modules",
        json.dumps(TARGET_MODULES),
        "--lora_type",
        setting.lora_type,
        "--train_on_inputs",
        bool_text(setting.train_on_inputs),
    ]
    if setting.precision:
        common.extend(("--precision", setting.precision))

    if setting.launcher == "model_parallel":
        return [
            sys.executable,
            str(ROOT / setting.train_script),
            *common,
            "--use_model_parallel",
            "True",
            "--device_map_strategy",
            "balanced",
            "--model_parallel_max_memory",
            max_memory,
            "--use_compile",
            "False",
        ]

    nproc = min(setting.nproc_per_node, len(gpus))
    if nproc < 1:
        raise ValueError("At least one GPU is required")
    return [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        str(ROOT / setting.train_script),
        *common,
    ]


def evaluation_command(
    setting: Table3Setting,
    base_model: str,
    output_dir: Path,
    run_name: str,
    dataset: str,
    batch_size: int,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / setting.eval_script),
        "--dataset",
        dataset,
        "--model",
        setting.model_tag,
        "--base_model",
        base_model,
        "--lora_weights",
        str(output_dir),
        "--lora_type",
        setting.lora_type,
        "--lora_alpha",
        str(setting.lora_alpha),
        "--rank",
        "32",
        "--adapter",
        "LoRA",
        "--run",
        run_name,
        "--batch_size",
        str(batch_size),
        "--load_dtype",
        "bf16",
    ]


def evaluate_all(
    setting: Table3Setting,
    base_model: str,
    output_dir: Path,
    run_name: str,
    gpus: list[str],
    env: dict[str, str],
    batch_size: int,
    dry_run: bool,
) -> None:
    datasets = [dataset for dataset, _, _ in DATASETS]

    for start in range(0, len(datasets), len(gpus)):
        wave = datasets[start : start + len(gpus)]
        processes = []
        for gpu, dataset in zip(gpus, wave):
            command = evaluation_command(
                setting, base_model, output_dir, run_name, dataset, batch_size
            )
            print(f"CUDA_VISIBLE_DEVICES={gpu} $ {command_text(command)}", flush=True)
            if dry_run:
                continue
            process_env = env.copy()
            process_env["CUDA_VISIBLE_DEVICES"] = gpu
            processes.append((dataset, subprocess.Popen(command, cwd=ROOT, env=process_env)))
        for dataset, process in processes:
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, dataset)


def main() -> None:
    args = parse_args()
    setting = SETTINGS[args.model]
    seed = setting.paper_seed if args.seed is None else args.seed
    base_model = args.base_model or setting.base_model
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one device")

    run_name = setting.run_name(seed)
    output_dir = args.run_root / "trained_models" / run_name
    experiment_dir = args.run_root / "experiment"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env["EXPERIMENT_DIR"] = str(experiment_dir)
    env["WANDB_DISABLED"] = "true"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if args.hf_cache:
        env["HF_HOME"] = str(args.hf_cache)
        env["HF_HUB_CACHE"] = str(args.hf_cache)
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    print(f"Model: {setting.label}")
    print(f"Seed: {seed}")
    print(f"Run: {run_name}")
    print(f"Output: {output_dir}")

    if args.phase in ("train", "both"):
        command = training_command(
            setting, base_model, seed, output_dir, gpus, args.model_parallel_max_memory
        )
        run_command(command, env, args.dry_run)

    if args.phase in ("eval", "both"):
        if not args.dry_run and not adapter_exists(output_dir):
            raise FileNotFoundError(f"Missing adapter weights: {output_dir}")
        evaluate_all(
            setting,
            base_model,
            output_dir,
            run_name,
            gpus,
            env,
            args.eval_batch_size,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
