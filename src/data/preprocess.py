import json
import os
import random
import re

from datasets import load_dataset as hf_load_dataset

DATASET_NAME = "keivalya/MedQuad-MedicalQnADataset"
OUTPUT_DIR = "data"


def load_dataset(name: str = DATASET_NAME):
    return hf_load_dataset(name, split="train")


def clean(dataset):
    def _normalize(example):
        example["Question"] = example["Question"].strip()
        example["Answer"] = re.sub(r"\s+", " ", example["Answer"]).strip()
        return example

    dataset = dataset.map(_normalize)
    dataset = dataset.filter(lambda ex: len(ex["Question"]) > 0 and len(ex["Answer"]) > 0)

    seen = set()
    records = []
    for example in dataset:
        key = (example["Question"], example["Answer"])
        if key in seen:
            continue
        seen.add(key)
        records.append(example)
    return records


def instruction_format(records):
    return [
        {
            "instruction": r["Question"],
            "input": "",
            "output": r["Answer"],
        }
        for r in records
    ]


def split(records, val_ratio=0.1, test_ratio=0.1, seed=42):
    records = records.copy()
    random.Random(seed).shuffle(records)
    n = len(records)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = records[:n_test]
    val = records[n_test:n_test + n_val]
    train = records[n_test + n_val:]
    return train, val, test


def save_json(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def run(output_dir: str = OUTPUT_DIR):
    dataset = load_dataset()
    records = clean(dataset)
    formatted = instruction_format(records)
    train, val, test = split(formatted)
    save_json(train, os.path.join(output_dir, "train.json"))
    save_json(val, os.path.join(output_dir, "val.json"))
    save_json(test, os.path.join(output_dir, "test.json"))
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    run()
