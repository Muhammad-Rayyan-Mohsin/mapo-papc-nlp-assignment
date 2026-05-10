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

## What's in here

A three-act research story:

| Version | Approach | Final train loss | Reward margin (peak) |
|---|---|---:|---:|
| v1 — Answer-equality scoring | Score candidates by `2·matches_pivot + 1·matches_gold` | 0.6932 | 0.0004 |
| v2 — NLLB faithful scoring | Paper-faithful `−CE(NLLB(Y_lang → Y_en)) / len` | 0.6929 | 0.0007 |
| **v3 — PAPC (our contribution)** | v2 + back-translated synthetic chosen for low-resource langs | **0.6878** | **0.011** |

PAPC produces a **16× larger reward margin** and a **26× larger train-loss reduction** than v2 — the metrics DPO directly optimizes.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Apple M4 Pro (MPS). CUDA + CPU also supported (set automatically).

## Inference

```bash
python inference.py --problem "What is 12 * 7?" --lang en
python inference.py --problem "12 * 7 ni ngapi?" --lang sw
```

Or open `notebooks/01_inference_demo.ipynb`.

## Training

The training script consumes `data/preferences.jsonl` (full file is ~120 MB and not committed — `data/sample_data.csv` shows the schema). To regenerate it, run the four-stage data pipeline documented in `mapo-repro/scripts/` of the original research repo, then:

```bash
python train.py --config config.yaml --preferences data/preferences.jsonl
```

Wall time at default settings (4 langs, K=3, 40 problems): ~50 min on M4 Pro.

## Method (one paragraph)

MAPO trains a multilingual math model with DPO using *language-alignment preferences* rather than human labels: for each problem we sample K candidate solutions per language, score them by `−CE(NLLB(candidate → English))` (how well the model's reasoning back-translates to the pivot answer), and build chosen/rejected pairs from that ordering. **PAPC** observes that low-resource languages (sw, bn) produce too few natural pairs because almost all candidates are wrong — so the back-translation signal collapses. PAPC injects synthetic *chosen* candidates by translating the correct English solution into the target language with NLLB, pairing them against the model's lowest-scoring native attempts. This keeps the preference distribution dense exactly where the base model is weakest.

## Results

See `results/baseline_metrics.json` vs `results/improved_metrics.json` for per-language MGSM accuracy. The headline DPO metrics (loss + reward margin) improve sharply with PAPC; the eval accuracy on a 25-problem-per-language MGSM slice is comparable to baseline at this compute scale — the proposed mechanism is what's being demonstrated, not a SOTA number.

## Files trail

- `config.yaml` — every hyperparameter; mirrors `mapo-repro/configs/config.py`
- `data/sample_data.csv` — 8 real preference pairs from training (zh examples)
- `results/training_log.csv` — DPO loss, rewards/margins, grad norm per step
- `checkpoints/mapo-adapter/` — LoRA adapter (q/k/v/o, r=16, α=32) + tokenizer
