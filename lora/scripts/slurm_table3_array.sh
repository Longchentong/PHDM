#!/usr/bin/env bash
#SBATCH --job-name=phdm_table3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/phdm_table3-%A_%a.out
#SBATCH --error=logs/phdm_table3-%A_%a.err

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT to a scratch output directory}
HF_HOME=${HF_HOME:?Set HF_HOME to a Hugging Face cache directory}

# One task per reported model; each run uses that model's paper seed.
MODELS=(llama13 gemma7 llama3 qwen25)
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
MODEL=${MODELS[$TASK_ID]}

mkdir -p "$ROOT/logs" "$RUN_ROOT"
cd "$ROOT"

ARGS=(
  --model "$MODEL"
  --run-root "$RUN_ROOT"
  --hf-cache "$HF_HOME"
  --gpus 0,1,2,3
  --offline
)

python experiments/run_table3.py "${ARGS[@]}"
