"""Policy loading + DPO/LoRA training + greedy evaluation generation.

Mirrors mapo-repro/scripts/03_train_dpo.py and 04_eval_mgsm.py:
  - ref_model=None (PEFT inserts the adapter; trl reuses the base as ref)
  - processing_class=tokenizer  (TRL 0.12 API)
  - dtype picked per device (fp16 on MPS, bf16 on CUDA, fp32 on CPU)
  - greedy decoding at eval time (matches the recorded MGSM run)
"""
from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from .dataset import load_preferences


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dtype(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def load_policy(base_model: str, adapter_path: str | None = None):
    device = _device()
    dtype = _dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(device)
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path).to(device)
    model.eval()
    return model, tokenizer


def train_dpo(cfg: dict, preferences_path: Path, output_dir: Path) -> None:
    device = _device()
    dtype = _dtype(device)
    base = cfg["models"]["base_model"]
    train = cfg["training"]
    lora = cfg["lora"]

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(device)

    peft_cfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["targets"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    ds = load_preferences(preferences_path)
    adapter_out = output_dir / "mapo-adapter"

    dpo_cfg = DPOConfig(
        output_dir=str(adapter_out),
        num_train_epochs=train["epochs"],
        per_device_train_batch_size=train["batch_size"],
        gradient_accumulation_steps=train["grad_accum"],
        learning_rate=train["lr"],
        beta=train["dpo_beta"],
        max_length=train["max_len"],
        max_prompt_length=train["max_prompt_len"],
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=cfg["seed"],
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)


@torch.no_grad()
def generate_answer(model, tokenizer, problem: str, gen_cfg: dict) -> str:
    """Greedy decode — matches the recorded MGSM evaluation run."""
    msgs = [
        {"role": "system", "content": "You are a careful math tutor."},
        {"role": "user", "content": (
            "Solve the following math problem step by step. "
            "End your answer with '#### <number>'.\n\n"
            f"Problem: {problem}\n\nSolution:"
        )},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    out = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=gen_cfg["max_new_eval"],
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()
