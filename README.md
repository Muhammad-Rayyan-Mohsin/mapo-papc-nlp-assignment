# MAPO Reproduction + PAPC Improvement

University NLP assignment. Laptop-scale reproduction of **MAPO — Multilingual Alignment via Preference Optimization** ([arXiv:2401.06838](https://arxiv.org/abs/2401.06838)) on `Qwen2.5-Math-1.5B-Instruct` + LoRA, plus a novel improvement: **PAPC (Pivot-Augmented Pair Construction)**.

## Repository layout

```
project-root/
├── README.md
├── requirements.txt
├── train.py             # DPO + LoRA training entrypoint
├── inference.py         # Generate an answer with the trained adapter
├── config.yaml          # All hyperparameters
├── data/
│   └── sample_data.csv  # 8 example preference pairs (audit trail)
├── notebooks/
│   └── 01_inference_demo.ipynb
├── src/
│   ├── model.py         # load_policy / train_dpo / generate_answer
│   ├── dataset.py       # preference-pair loading + PAPC helpers
│   └── utils.py         # seeding, answer extraction, jsonl I/O
├── results/
│   ├── baseline_metrics.json   # MGSM accuracy, base model (no DPO)
│   ├── improved_metrics.json   # MGSM accuracy, MAPO + PAPC
│   └── training_log.csv        # Per-step DPO metrics
└── checkpoints/
    └── mapo-adapter/    # LoRA adapter (~17 MB safetensors)
```

## Setup at this scale

The paper uses MathOctopus-7B with full fine-tuning, thousands of GSM8K problems, 20 candidates per problem, 3 rounds of iterative DPO, and 250 eval problems × 10 languages on a multi-GPU cluster.

This reproduction runs on an Apple M4 Pro laptop: Qwen2.5-Math-1.5B + LoRA (r=16, α=32, q/k/v/o), 40 training problems, K=3 candidates per (problem, language), 1 DPO round, 25 eval problems × 4 languages (en, zh, sw, bn).

## Three-act story

| Version | Approach | Final train loss | Reward margin (peak) |
|---|---|---:|---:|
| v1 — Answer-equality scoring | Score by `2·matches_pivot + 1·matches_gold` | 0.6932 | 0.0004 |
| v2 — NLLB faithful scoring | Paper-faithful `−CE(NLLB(Y_lang → Y_en)) / len` | 0.6929 | 0.0007 |
| **v3 — PAPC (our contribution)** | v2 + back-translated synthetic chosen for low-resource langs | **0.6878** | **0.011** |

PAPC shifts the loss reduction from 0.0003 (v2 vs ln 2) to 0.0054 — roughly an **18× larger reduction** — and the reward margin grows ~**16×**. These are the quantities DPO is directly optimizing.

## What PAPC actually fixes

v2 (faithful reproduction) revealed a structural problem the paper does not flag: low-resource languages produce almost no preference pairs because the base model can't generate a correct candidate, and the paper's "chosen must be correct" gate stays shut.

- Chinese: 41 natural pairs from 40 problems
- Bengali: 13 pairs
- Swahili: **2 pairs**

So a method designed to fix multilingual gaps was, in practice, training almost entirely on Chinese. **PAPC** detects this (pairs < 30 for a language) and injects up to 8 synthetic chosen by NLLB-back-translating the trusted English solution into the target language. Final counts: Swahili 2 → 10, Bengali 13 → 21, total 56 → 72.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Device handling is automatic (MPS / CUDA / CPU).

## Inference

```bash
python inference.py --problem "What is 12 * 7?" --lang en
python inference.py --problem "12 * 7 ni ngapi?" --lang sw
```

Or open `notebooks/01_inference_demo.ipynb`. Decoding is greedy, matching the evaluation run.

## Training

`train.py` consumes `data/preferences.jsonl`; the full file isn't committed (`data/sample_data.csv` shows the schema). To regenerate it, run the four staged scripts (`01_translate.py` → `02_build_preferences.py`) from the research repo, then:

```bash
python train.py --config config.yaml --preferences data/preferences.jsonl
```

Wall time at default settings: ~50 minutes on M4 Pro.

## Results — honest reporting

| Lang | Baseline MGSM acc | MAPO + PAPC MGSM acc |
|---|---:|---:|
| en | 0.88 | 0.88 |
| zh | 0.64 | 0.64 |
| sw | 0.04 | 0.04 |
| bn | 0.08 | 0.04 |

**MGSM accuracy does not move** at this scale. The DPO training dynamics (loss + reward margin) improve sharply with PAPC, but a 0.0054 loss reduction across 9 optimizer steps cannot propagate into per-question accuracy changes — the weight updates are too small. The 25-problem-per-language eval also has a binomial standard error around 10 pp for low-accuracy languages, so any small real signal is below the noise floor.

The takeaway: PAPC shifts the bottleneck from data starvation to optimizer budget — that's progress at the mechanism level, not a SOTA accuracy claim.

See `results/baseline_metrics.json` and `results/improved_metrics.json` for the raw numbers and `results/training_log.csv` for per-step DPO metrics.

## Files trail

- `config.yaml` — every hyperparameter; mirrors `mapo-repro/configs/config.py`
- `data/sample_data.csv` — 8 real preference pairs from training (zh examples)
- `results/training_log.csv` — DPO loss, rewards/margins, grad norm per step
- `checkpoints/mapo-adapter/` — LoRA adapter + tokenizer artifacts
