import json

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
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


def train(train_path="data/train.json", val_path="data/val.json", output_dir="adapters/lora"):
    model_cfg, training_cfg = load_configs()
    use_cuda = torch.cuda.is_available()

    train_dataset = load_split(train_path)
    eval_dataset = load_split(val_path)

    lora_config = LoraConfig(
        r=training_cfg["lora_r"],
        lora_alpha=training_cfg["lora_alpha"],
        lora_dropout=training_cfg["lora_dropout"],
        target_modules=training_cfg["target_modules"],
        task_type="CAUSAL_LM",
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
        fp16=use_cuda,
        model_init_kwargs={"dtype": torch.float16 if use_cuda else torch.float32},
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model_cfg["base_model"],
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(output_dir)

    return trainer
