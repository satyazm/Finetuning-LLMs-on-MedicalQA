import json
import os
import random
import time

import pandas as pd
import torch
from evaluate import load as load_metric
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def build_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate(model, tokenizer, instruction, max_new_tokens=256, device="cuda"):
    prompt = build_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - start

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip(), latency


def sample_examples(examples, n=200, seed=42):
    return random.Random(seed).sample(examples, min(n, len(examples)))


def evaluate_examples(model, tokenizer, examples, max_new_tokens=256, device="cuda"):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = []
    for ex in examples:
        prediction, latency = generate(model, tokenizer, ex["instruction"], max_new_tokens, device)
        rows.append({
            "instruction": ex["instruction"],
            "reference": ex["output"],
            "prediction": prediction,
            "latency_sec": latency,
        })

    df = pd.DataFrame(rows)
    rouge = load_metric("rouge")
    scores = rouge.compute(predictions=df["prediction"].tolist(), references=df["reference"].tolist())

    summary = dict(scores)
    summary["mean_latency_sec"] = float(df["latency_sec"].mean())
    summary["gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    return df, summary


def save_results(df, summary, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    summary_path = csv_path.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def run_baseline(model_name, test_path, output_csv, n_samples=200, max_new_tokens=256, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(test_path) as f:
        test_examples = json.load(f)

    examples = sample_examples(test_examples, n=n_samples)
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    df, summary = evaluate_examples(model, tokenizer, examples, max_new_tokens, device)
    save_results(df, summary, output_csv)
    print(json.dumps(summary, indent=2))
    return df, summary
