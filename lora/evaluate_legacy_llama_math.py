import copy
import json
import os
import re
import sys
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig, LlamaTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_dtype(dtype_name: str):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported load dtype: {dtype_name}")


def generate_prompt(instruction, input_text=None):
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def parse_output(output: str):
    if "### Response:" in output:
        return output.split("### Response:", 1)[1].strip()
    return output.strip()


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    try:
        return float(pred[-1])
    except ValueError:
        return float("inf")


def extract_answer_letter(sentence: str) -> str:
    sentence = sentence.strip()
    patterns = [
        r"\\boxed\{\\text\{\(?([A-E])\)?\}\}",
        r"\\boxed\{\(?([A-E])\)?\}",
        r"\(([A-E])\)",
        r"answer is\s+([A-E])",
        r"correct answer is\s+([A-E])",
        r"\b([A-E])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, sentence, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return "N/A"


def result_path(args):
    experiment_dir = os.environ.get("EXPERIMENT_DIR", "experiment")
    os.makedirs(experiment_dir, exist_ok=True)
    return os.path.join(
        experiment_dir,
        f"{args.model}-{args.adapter}-{args.dataset}-{args.rank}-"
        f"{args.lora_type}-{args.lora_alpha}-{args.run}.json",
    )


def load_data(dataset):
    file_path = ROOT / "dataset" / dataset / "test.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    with file_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_model(args):
    load_dtype = get_torch_dtype(args.load_dtype)
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left" if args.batch_size > 1 else "right"
    tokenizer.pad_token_id = tokenizer.pad_token_id or 0

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_dtype,
        device_map={"": 0} if device == "cuda" else {"": device},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        model,
        args.lora_weights,
        torch_dtype=load_dtype,
        device_map={"": 0} if device == "cuda" else {"": device},
    )
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return tokenizer, model


def evaluate_one(tokenizer, model, instruction, input_text, max_new_tokens):
    prompt = generate_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_config = GenerationConfig(
        num_beams=4,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=False)
    return parse_output(output)


def evaluate_batch(tokenizer, model, batch_data, max_new_tokens):
    prompts = [
        generate_prompt(data.get("instruction"), data.get("input"))
        for data in batch_data
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    generation_config = GenerationConfig(
        num_beams=4,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    return [
        parse_output(tokenizer.decode(sequence, skip_special_tokens=False))
        for sequence in generation_output.sequences
    ]


def flush(path, output_data):
    with open(path, "w") as f:
        json.dump(output_data, f, indent=4)


def main():
    args = parse_args()
    dataset = load_data(args.dataset)
    save_file = result_path(args)
    output_data = []
    correct = 0
    flush(save_file, output_data)

    tokenizer, model = load_model(args)
    miss = 0.001
    pbar = tqdm(total=len(dataset))
    batch_size = max(1, args.batch_size)

    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_data = dataset[batch_start:batch_end]
        if batch_size == 1:
            batch_outputs = [
                evaluate_one(
                    tokenizer,
                    model,
                    batch_data[0].get("instruction"),
                    batch_data[0].get("input"),
                    args.max_new_tokens,
                )
            ]
        else:
            batch_outputs = evaluate_batch(tokenizer, model, batch_data, args.max_new_tokens)

        for offset, (data, outputs) in enumerate(zip(batch_data, batch_outputs)):
            idx = batch_start + offset
            label = data.get("answer")
            flag = False
            if args.dataset.lower() == "aqua":
                predict = extract_answer_letter(outputs)
                if label == predict:
                    correct += 1
                    flag = True
            else:
                if isinstance(label, str):
                    label = float(label)
                predict = extract_answer_number(outputs)
                if abs(label - predict) <= miss:
                    correct += 1
                    flag = True

            new_data = copy.deepcopy(data)
            new_data["output_pred"] = outputs
            new_data["pred"] = predict
            new_data["flag"] = flag
            new_data["accuracy"] = correct / (idx + 1)
            output_data.append(new_data)
            print(" ")
            print("---------------")
            print(outputs)
            print("prediction:", predict)
            print("label:", label)
            print("---------------")
            print(f"\rtest:{idx + 1}/{len(dataset)} | accuracy {correct}  {correct / (idx + 1)}")
            pbar.update(1)
        flush(save_file, output_data)

    pbar.close()
    flush(save_file, output_data)
    print("\n\ntest finished")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "AQuA", "SVAMP", "mawps"], required=True)
    parser.add_argument("--model", choices=["LLaMA-13B"], required=True)
    parser.add_argument("--adapter", choices=["LoRA"], required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--lora_weights", required=True)
    parser.add_argument("--lora_type", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--lora_alpha", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--load_dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    main()
