import os
import sys
from pathlib import Path
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, LlamaTokenizer

from adapter_checkpoint_trainer import PeftAdapterCheckpointTrainer


class ModelParallelCausalLMTrainer(PeftAdapterCheckpointTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction="sum" if num_items_in_batch is not None else "mean",
        )
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        if num_items_in_batch is not None:
            if not torch.is_tensor(num_items_in_batch):
                num_items_in_batch = torch.tensor(
                    num_items_in_batch, device=loss.device, dtype=loss.dtype
                )
            else:
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch

        return (loss, outputs) if return_outputs else loss


def train(
    base_model: str = "",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    adapter_name: str = "lora",
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    seed: int = 42,
    use_gradient_checkpointing: bool = False,
    eval_step: int = 200,
    save_step: int = 200,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_type: str = "std",
    use_dora: bool = False,
    target_modules: List[str] = None,
    train_on_inputs: bool = True,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    use_model_parallel: bool = False,
    device_map_strategy: str = "balanced",
    model_parallel_max_memory: str = "",
    use_compile: bool = True,
):
    print(
        f"Finetuning legacy LLaMA with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"seed: {seed}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_type: {lora_type}\n"
        f"use_dora: {use_dora}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"use_model_parallel: {use_model_parallel}\n"
        f"device_map_strategy: {device_map_strategy}\n"
        f"model_parallel_max_memory: {model_parallel_max_memory}\n"
        f"use_compile: {use_compile}\n"
    )

    assert base_model, "Please specify a --base_model"
    transformers.set_seed(seed)
    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    elif use_model_parallel and torch.cuda.device_count() > 1:
        device_map = device_map_strategy
    else:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    max_memory = None
    if use_model_parallel and model_parallel_max_memory:
        max_memory = {
            gpu_id: model_parallel_max_memory for gpu_id in range(torch.cuda.device_count())
        }

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.pad_token_id or 0

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            add_eos_token
            and result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt[
                "labels"
            ][user_prompt_len:]
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=use_gradient_checkpointing
    )
    config = LoraConfig(
        use_dora=use_dora,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        lora_type=lora_type,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint and not os.path.isdir(resume_from_checkpoint):
        raise ValueError(f"Checkpoint directory not found: {resume_from_checkpoint}")

    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = train_val["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer_cls = ModelParallelCausalLMTrainer if use_model_parallel else transformers.Trainer
    trainer = trainer_cls(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            save_safetensors=False,
            label_names=["labels"],
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            seed=seed,
            data_seed=seed,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if use_compile and torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir, safe_serialization=False)


def generate_prompt(data_point):
    instruction = data_point["instruction"]
    input_text = data_point.get("input", "")
    output_text = data_point["output"]

    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output_text}"
    )


if __name__ == "__main__":
    fire.Fire(train)
