import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_configs(model_config_path="configs/model.yaml", training_config_path="configs/training.yaml"):
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)
    with open(training_config_path) as f:
        training_cfg = yaml.safe_load(f)
    return model_cfg, training_cfg


def to_prompt_completion(records):
    return [
        {
            "prompt": f"### Instruction:\n{r['instruction']}\n\n### Response:\n",
            "completion": r["output"],
        }
        for r in records
    ]


def load_split(path):
    with open(path) as f:
        records = json.load(f)
    return Dataset.from_list(to_prompt_completion(records))


def train(train_path="data/train.json", val_path="data/val.json", output_dir="adapters/qlora", resume_from_checkpoint=None):
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU (bitsandbytes 4-bit quantization is not supported on CPU).")

    model_cfg, training_cfg = load_configs()

    train_dataset = load_split(train_path)
    eval_dataset = load_split(val_path)

    lora_config = LoraConfig(
        r=training_cfg["lora_r"],
        lora_alpha=training_cfg["lora_alpha"],
        lora_dropout=training_cfg["lora_dropout"],
        target_modules=training_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=model_cfg["max_seq_length"],
        num_train_epochs=training_cfg["num_epochs"],
        per_device_train_batch_size=training_cfg["batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        logging_steps=training_cfg["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=training_cfg.get("report_to", "none"),
        bf16=True,
        fp16=False,
        model_init_kwargs={"dtype": torch.bfloat16, "device_map": {"": 0}},
        gradient_checkpointing_kwargs={"use_reentrant": False},
        loss_type="nll",
    )

    trainer = SFTTrainer(
        model=model_cfg["base_model"],
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        quantization_config=bnb_config,
    )

    torch.cuda.reset_peak_memory_stats()

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak training GPU memory: {peak_memory_gb:.2f} GB")

    trainer.save_model(output_dir)

    return trainer
