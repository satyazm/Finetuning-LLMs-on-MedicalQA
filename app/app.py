import time

import gradio as gr
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("configs/model.yaml") as f:
    MODEL_CFG = yaml.safe_load(f)

BASE_MODEL = MODEL_CFG["base_model"]
MAX_NEW_TOKENS = 128

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

print(f"Loading {BASE_MODEL} on {DEVICE} ({DTYPE})...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=DTYPE)
model = PeftModel.from_pretrained(model, "adapters/lora", adapter_name="lora")
model.load_adapter("adapters/qlora", adapter_name="qlora")
model = model.to(DEVICE)
model.eval()
print("Ready.")


def build_prompt(instruction):
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_answer(variant, question):
    if not question.strip():
        return "", ""

    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        if variant == "Base":
            with model.disable_adapter():
                output_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
        else:
            model.set_adapter("lora" if variant == "LoRA" else "qlora")
            output_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
    latency = time.perf_counter() - start

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return answer, f"{latency:.2f}s"


demo = gr.Interface(
    fn=generate_answer,
    inputs=[
        gr.Radio(["Base", "LoRA", "QLoRA"], value="LoRA", label="Model"),
        gr.Textbox(lines=2, label="Question", placeholder="e.g. What are the symptoms of asthma?"),
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Latency"),
    ],
    examples=[
        ["Base", "What are the symptoms of asthma?"],
        ["LoRA", "What are the symptoms of asthma?"],
        ["QLoRA", "What are the symptoms of asthma?"],
    ],
    title="MedQuAD PEFT Demo",
    description="Compare zero-shot, LoRA, and QLoRA fine-tuned Qwen2.5-1.5B-Instruct on medical Q&A.",
)

if __name__ == "__main__":
    demo.launch()
