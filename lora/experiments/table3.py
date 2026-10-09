from __future__ import annotations

from dataclasses import dataclass


TARGET_MODULES = ("q_proj", "v_proj", "k_proj", "up_proj", "down_proj")
DATASETS = (
    ("mawps", "MAWPS", 238),
    ("SVAMP", "SVAMP", 1000),
    ("gsm8k", "GSM8K", 1319),
    ("AQuA", "AQuA", 254),
)
TOTAL_EXAMPLES = sum(size for _, _, size in DATASETS)


@dataclass(frozen=True)
class Table3Setting:
    key: str
    label: str
    base_model: str
    model_tag: str
    train_script: str
    eval_script: str
    data_file: str
    run_prefix: str
    lora_type: str
    lora_alpha: int
    lora_dropout: float
    epochs: int
    learning_rate: float
    micro_batch_size: int
    train_on_inputs: bool
    paper_seed: int
    launcher: str
    nproc_per_node: int
    gradient_checkpointing: bool = False
    precision: str | None = None

    def run_name(self, seed: int) -> str:
        return f"{self.run_prefix}-seed{seed}"


SETTINGS = {
    "llama13": Table3Setting(
        key="llama13",
        label="LLaMA-13B",
        base_model="huggyllama/llama-13b",
        model_tag="LLaMA-13B",
        train_script="finetune_legacy_llama.py",
        eval_script="evaluate_legacy_llama_math.py",
        data_file="ft-training_set/math_10k.json",
        run_prefix="llama-13b-phdm-c0.5-r32-math",
        lora_type="phdm-0.5",
        lora_alpha=256,
        lora_dropout=0.05,
        epochs=3,
        learning_rate=3e-4,
        micro_batch_size=1,
        train_on_inputs=True,
        paper_seed=42,
        launcher="model_parallel",
        nproc_per_node=4,
    ),
    "gemma7": Table3Setting(
        key="gemma7",
        label="Gemma-7B",
        base_model="google/gemma-7b",
        model_tag="gemma-7b",
        train_script="finetune_gemma7b.py",
        eval_script="evaluate.py",
        data_file="ft-training_set/math_10k.json",
        run_prefix="gemma-7b-phdm-c0.5-r32-math",
        lora_type="phdm-0.5",
        lora_alpha=256,
        lora_dropout=0.05,
        epochs=3,
        learning_rate=3e-4,
        micro_batch_size=1,
        train_on_inputs=True,
        paper_seed=42,
        launcher="model_parallel",
        nproc_per_node=4,
        gradient_checkpointing=True,
    ),
    "llama3": Table3Setting(
        key="llama3",
        label="LLaMA3-8B",
        base_model="meta-llama/Meta-Llama-3-8B-Instruct",
        model_tag="Meta-Llama-3-8B-Instruct",
        train_script="finetune_llama3.py",
        eval_script="evaluate.py",
        data_file="ft-training_set/math_10k_answer_only.json",
        run_prefix="llama3-8b-phdm-answeronly-c1.0-a128-e5-lr2e-4-d0",
        lora_type="phdm-1.0",
        lora_alpha=128,
        lora_dropout=0.0,
        epochs=5,
        learning_rate=2e-4,
        micro_batch_size=1,
        train_on_inputs=False,
        paper_seed=42,
        launcher="torchrun",
        nproc_per_node=4,
    ),
    "qwen25": Table3Setting(
        key="qwen25",
        label="Qwen2.5-7B",
        base_model="Qwen/Qwen2.5-7B-Instruct-1M",
        model_tag="Qwen2.5-7B-Instruct-1M",
        train_script="finetune_qwen.py",
        eval_script="evaluate.py",
        data_file="ft-training_set/math_10k_answer_only.json",
        run_prefix="qwen25-1m-phdm-kconst-c1.0-a128-e3-lr3e-4-d0",
        lora_type="phdm_kconst-1.0",
        lora_alpha=128,
        lora_dropout=0.0,
        epochs=3,
        learning_rate=3e-4,
        micro_batch_size=1,
        train_on_inputs=False,
        paper_seed=456,
        launcher="torchrun",
        nproc_per_node=2,
        precision="bf16",
    ),
}
