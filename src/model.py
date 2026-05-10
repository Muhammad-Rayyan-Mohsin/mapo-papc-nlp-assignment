"""Policy loading + DPO/LoRA training + generation.

Kept intentionally small — the goal is a clean, reviewable surface that
matches the staged scripts in mapo-repro/scripts/ (03_train_dpo.py and
04_eval_mgsm.py). For full reproducibility (data prep -> preferences ->
train -> eval) see README.md.
"""
from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from .dataset import load_preferences


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_policy(base_model: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
    ).to(_device())
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def train_dpo(cfg: dict, preferences_path: Path, output_dir: Path) -> None:
    base = cfg["models"]["base_model"]
    train = cfg["training"]
    lora = cfg["lora"]

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    ref = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)

    peft_cfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["targets"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    ds = load_preferences(preferences_path)

    dpo_cfg = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train["batch_size"],
        gradient_accumulation_steps=train["grad_accum"],
        num_train_epochs=train["epochs"],
        learning_rate=train["lr"],
        beta=train["dpo_beta"],
        max_prompt_length=train["max_prompt_len"],
        max_length=train["max_len"],
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=cfg["seed"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref,
        args=dpo_cfg,
        train_dataset=ds,
        tokenizer=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "mapo-adapter"))


@torch.no_grad()
def generate_answer(model, tokenizer, problem: str, gen_cfg: dict) -> str:
    prompt = (
        "<|im_start|>system\nYou are a careful math tutor.<|im_end|>\n"
        "<|im_start|>user\nSolve the following math problem step by step. "
        "End your answer with '#### <number>'.\n\n"
        f"Problem: {problem}\n\nSolution:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=gen_cfg["max_new_eval"],
        do_sample=True,
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()
