# Knowledge-Aware MARL for EM Financial Stability

> **From Risk Detection to Risk Mitigation: A Knowledge-Aware Multi-Agent Framework for Dynamic Financial Stability in Emerging Markets**
>
> Yixue Hao · Pu Han · Lianxing Min
> *Submitted to Knowledge-Based Systems (Elsevier), 2025*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Anonymous repo](https://img.shields.io/badge/anonymous-4open.science-lightgrey.svg)](https://anonymous.4open.science/r/em-fin-stability-5BD3/)

---

## Overview

This repository contains the full implementation of **KA-MARL**, a knowledge-aware
multi-agent reinforcement learning framework that closes the gap between crisis
*detection* and crisis *mitigation* in emerging-market (EM) economies.

The framework has three stages:

| Stage | Module | Role |
|-------|--------|------|
| **1 — Detection** | `detection.py` | Four calibrated binary crisis detectors → six-class posterior **π**_t |
| **2 — Retrieval** | `analogy_engine.py`, `case_memory.py` | Crisis-type-conditional Mahalanobis retrieval → context vector **φ**_t |
| **3 — Mitigation** | `mitigation.py` | Three-institution TD3 actors with case-coherence regulariser and CBF projection |

**Key results** (89-case library, 25 EM economies, 1990–2024, seeds {42, 1729, 2718, 3141, 8128}):

- AUC-PR **0.873** (+4.2 pp over LSTM baseline)
- Recall@1 **0.762** (3.6× vs. random)
- Mean output-loss reduction **38.4 %** (Cohen's *d* = 8.20 vs. no-retrieval)
- Synthetic-control treatment effect **+2.74 pp** GDP gap \[1.13, 4.41\]

---

## Repository Structure

```
.
├── case_memory.py       # CrisisCase schema, CaseLibrary (SHA-256 verified),
│                        #   CaseSignatureEncoder (SupCon loss, 32-dim signatures)
├── analogy_engine.py    # Conditional Mahalanobis retrieval (W_τ = L_τL_τᵀ + γI),
│                        #   CaseContextEncoder, InfoNCE training
├── detection.py         # Four binary detectors (Banking / Currency / Sovereign / Twin)
│                        #   + CoordinatorRouter, temperature calibration
├── coordinator.py       # Crisis-type posterior routing, authority-graph partitioning
├── mitigation.py        # CaseAugmentedActor (TD3), case-coherence regulariser,
│                        #   CBF barriers h₁–h₄, QP safe-set projection
├── training.py          # Five-phase sequential training loop (Phases A → D + E)
├── data_pipeline.py     # EM data ingestion, Laeven–Valencia labels, VAR simulator
├── evaluation.py        # AUC-PR/ROC, Recall@K, ECE, MRR, synthetic-control estimator
├── experiments.py       # Ablation runner (no_cbf / no_coord / no_coh / no_retr)
└── tests.py             # Unit and integration tests (pytest)
```

---

## Installation

```bash
# 1. Clone (or download from the anonymous link above)
git clone https://anonymous.4open.science/r/em-fin-stability-5BD3/
cd em-fin-stability-5BD3

# 2. Create environment (Python 3.10 or 3.11 recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.1 | Neural networks, SupCon loss, TD3 actors |
| `numpy` | ≥ 1.26 | Array operations |
| `scipy` | ≥ 1.12 | QP projection (CBF), PCHIP trajectories |
| `pandas` | ≥ 2.1 | EM macro time-series |
| `scikit-learn` | ≥ 1.4 | Calibration, evaluation metrics |
| `cvxpy` | ≥ 1.4 | CBF quadratic-program solver |
| `statsmodels` | ≥ 0.14 | VAR simulator, synthetic control |
| `matplotlib` | ≥ 3.8 | Figure generation scripts |

---

## Quick Start

### 1 — Build the case library

```python
from case_memory import CaseLibrary

lib = CaseLibrary.from_laeven_valencia(
    lv_path="data/laeven_valencia_2020.xlsx",
    macro_path="data/em_macro_1990_2024.csv",
)
lib.verify_integrity()          # SHA-256 chain check
print(f"Library: {len(lib)} cases, integrity OK")
```

### 2 — Run the full training pipeline

```python
from training import TrainingPipeline

pipeline = TrainingPipeline(
    case_library=lib,
    seeds=[42, 1729, 2718, 3141, 8128],
    device="cuda",               # or "cpu"
)
pipeline.run()                   # Phases A → B1/B2 → C → D
```

### 3 — Evaluate

```python
from evaluation import Evaluator

ev = Evaluator(pipeline.trained_models, lib.test_partition())
results = ev.run_all()
print(results.summary())
# AUC-PR: 0.873 | AUC-ROC: 0.941 | Recall@1: 0.762 | ECE: 5.6%
```

### 4 — Run ablations

```python
from experiments import AblationSuite

suite = AblationSuite(pipeline)
suite.run(["no_cbf", "no_coord", "no_coh", "no_retr"])
suite.export_latex("results/ablation_table.tex")
```

---

## Module Reference

### `case_memory.py` — Knowledge Base

```
CrisisCase          Dataclass with 7 field groups:
                      identification · trajectory (X_pre, X_post)
                      economic outcome · policy timeline
                      authority snapshot · provenance · reproducibility
CaseLibrary         SHA-256 content-verified container of CrisisCase objects
CaseSignatureEncoder  Convolutional-attention network → 32-dim unit-norm
                      signature trained with Supervised Contrastive loss
```

### `analogy_engine.py` — Retrieval Engine

```
ConditionalMahalanobis  W_τ = L_τ L_τᵀ + γI per crisis type τ
                          s(z_q, z_c | π) = Σ_τ π_τ · z_qᵀ W_τ z_c
                          Training: InfoNCE loss
CaseContextEncoder      Assembles 58-dim feature vectors
                          (32-dim signature + 24-dim policy + 2-dim outcome)
                          → attention pooling → linear projection → φ
```

### `mitigation.py` — Safe Policy Layer

```
CaseAugmentedActor  TD3 actor consuming (x_t, φ_t, π_t)
CaseCoherence       L_coh = 1 − cosine(a, â)  where â is the
                      retrieval-weighted, outcome-weighted target
CBFProjection       Four barriers h₁–h₄ (FX floor, capital floor,
                      sovereign capacity, fiscal space)
                      QP projection with γ_CBF = 0.7
```

### `training.py` — Training Phases

| Phase | Trains | Depends on |
|-------|--------|-----------|
| A | Four binary detectors | Case library labels |
| B1 | CoordinatorRouter | Phase A outputs |
| B2 | CaseSignatureEncoder (SupCon) | Laeven–Valencia type labels |
| C | Conditional Mahalanobis + CaseContextEncoder | Phase B2 signatures |
| D | Three-institution TD3 + CBF | Phases B1, C |
| E *(optional)* | Joint fine-tune | Phase D checkpoint |

---

## Reproducing Paper Results

All tables and figures in the paper can be reproduced from a single script:

```bash
# Full reproduction (≈ 4 h on a single A100 GPU)
python experiments.py --reproduce-all --seeds 42 1729 2718 3141 8128

# Figures only (requires pre-trained checkpoints in ./checkpoints/)
python experiments.py --figures-only

# Single seed fast check (≈ 25 min)
python experiments.py --seed 42 --fast
```

Figure generation scripts:

```bash
python fig05_calibration.py   # Reliability diagrams
python fig06_ablation.py      # Waterfall + radar + Cohen's d heatmap
python fig07_cases.py         # End-to-end case traces (Korea / Argentina / Lebanon)
python fig08_sc.py            # Synthetic-control forest plots
```

---

## Tests

```bash
pytest tests.py -v
```

Key test suites:

| Test class | Checks |
|-----------|--------|
| `TestCaseLibrary` | SHA-256 integrity, case retrieval, schema validation |
| `TestConditionalMahalanobis` | PSD guarantee, InfoNCE gradient flow |
| `TestCBFProjection` | Barrier feasibility, QP convergence, h₁–h₄ correctness |
| `TestTrainingPipeline` | Phase ordering, checkpoint save/load |
| `TestEvaluator` | AUC-PR/ROC, ECE, synthetic-control point estimate |

---

## Data

The macro panel (25 EM economies, 1990–2024, quarterly) and
Laeven–Valencia (2020) crisis labels are bundled under `data/` in the
anonymous repository. The raw sources are:

- **IMF World Economic Outlook** — GDP gap, CA balance, fiscal space
- **BIS** — credit gap, FX reserves, debt/GDP
- **Laeven & Valencia (2020)** — banking, currency, sovereign crisis dates

The `data_pipeline.py` module documents all cleaning steps and merges.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Anonymous submission. Author identities will be disclosed upon acceptance.*
