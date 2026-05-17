# From Risk Detection to Risk Mitigation

**A Multi-Agent Approach for Dynamic Financial Stability in Emerging
Markets**

Reference implementation accompanying the manuscript submitted to
*Expert Systems with Applications* (Elsevier). The codebase is
currently blinded for peer review; authorship details will be added at
acceptance.

**Software version: 0.2.0** (camera-ready ESWA revision).
See `CITATION.cff` and the *What changed in v0.2.0* section below for a
summary of revisions since the initial submission.

---

## Overview

This repository implements a three-stage pipeline for emerging-market
(EM) financial-stability monitoring and response:

1. **Detection.** A heterogeneous ensemble of four crisis detectors —
   conditional-entropy change-point detection (Kraevskiy et al., 2024),
   a variable-selection LSTM with GLU-gated GRN blocks (Aquilina et
   al., 2025), a sentence-transformer sentiment detector, and a
   temporal graph-attention network for cross-border contagion. The
   network detector accepts either real BIS LBS bilateral exposures
   (`BISBilateralProvider`, v0.2.0) or a uniform fallback adjacency.

2. **Coordination.** A *disagreement-aware* coordinator that treats the
   pattern of inter-detector disagreement as an informative signal
   rather than noise to suppress. A learned router conditions a
   regime-switching aggregator on a 32-dimensional disagreement
   embedding, supervised against Laeven-Valencia's multi-dimensional
   crisis-type flags (banking / currency / sovereign / twin / triple).
   MC-Dropout at inference time produces epistemic-uncertainty bands
   (v0.2.0).

3. **Mitigation.** An *authority-constrained* multi-agent RL policy
   whose joint action space is hard-constrained by the empirical
   policy-authority graph of each country (central bank → policy rate /
   FX intervention / reserve requirement; financial supervisor →
   capital adequacy / LTV cap / countercyclical buffer; ministry of
   finance → fiscal stance / debt issuance). A control-barrier-function
   (CBF) safety layer projects proposed actions onto the safe set.

### Three contributions

| # | Contribution | Where in the code |
|---|---|---|
| 1 | Detector disagreement as information, not noise. | `src/em_fin_stability/coordinator.py` |
| 2 | Authority-constrained MARL with CBF safety projection. | `src/em_fin_stability/mitigation_agents.py` |
| 3 | Counterfactual welfare-loss avoided on historical EM crises with Abadie placebo inference. | `src/em_fin_stability/evaluation.py::CounterfactualWelfare` |

The five historical EM episodes used for contribution #3:

| Country | Onset | End | Type |
|---|---|---|---|
| Türkiye | 2018 Q3 | 2019 | Currency |
| Argentina | 2018 Q2 | 2019 | Twin (currency + sovereign) |
| Sri Lanka | 2022 Q2 | 2023 | Twin (currency + sovereign) |
| Lebanon | 2019 Q4 | ongoing | Triple (banking + currency + sovereign) |
| Ghana | 2022 Q4 | 2023 | Sovereign |

---

## What changed in v0.2.0

This revision addresses reviewer comments from the first submission round
and incorporates feedback from the ten-paper extraction report covering
recent detector / coordinator / MARL methodology. The complete change list:

**Data pipeline.** 28-economy panel (24 MSCI EM members as of January 2025
including Kuwait and the UAE; plus the four non-MSCI post-2017 crisis
subjects ARG, LBN, LKA, GHA). The country roster is split into the canonical
`MSCI_EM_COUNTRIES` and `NON_MSCI_STUDY_COUNTRIES` constants, with the
backward-compatible `EM_COUNTRIES` alias preserving the 28-country union.
The Hodrick-Prescott one-sided filter remains the default, but a Hamilton
(2018) regression-based detrender is now available via
`filter_method: "hamilton"` in the data-build config — Hamilton is a vocal
HP critic and we now report the headline detection results under both
filters as a robustness check. `POST_2017_CRISES` entries are now 5-tuples
`(iso3, year, quarter, end_year, type)` (was 4-tuples), and a 16-predictor
Bluwstein-faithful specification (`BLUWSTEIN_PREDICTORS_FULL`) plus its
12-predictor EM-coverage subset (`BLUWSTEIN_PREDICTORS_EM`) are exposed for
use by the baseline factory.

**Detectors.** RNN/VSN: `n_variables` is 12 (the panel feature width:
7 macro state vars + 5 engineered), down from a wider input in v0.1.0;
`n_temporal_features` is 4 (was 5; we dropped a constant
quarter-of-year column that contributed no signal); GLU gating in the
gated-residual-network (GRN) blocks is on by default. The
`NetworkContagionDetector` now consumes a `BilateralAdjacencyProvider`
(new ABC) instead of an internal uniform default; the v0.2.0
`BISBilateralProvider` reads quarterly LBS bilateral exposures from
Parquet and falls back to `UniformAdjacencyProvider` if the file is
unavailable. Every detector now implements
`predict_with_uncertainty(...)` returning MC-Dropout (neural) or
bootstrap (entropy / sentiment) uncertainty alongside the score.

**Coordinator.** The `DisagreementFeaturizer` has 32 features by
default (formula: `3K + K(K-1)/2 + 6 + K + K` for `K=4` detectors),
unchanged from the v0.1.0 mathematical specification but now with a
runtime shape assertion. `pre_horizon_quarters: 12` is now supported
(Bluwstein-faithful 3-year horizon); `replicate.yaml` uses this with
`crisis_label_col: "crisis_pre_3y"` for direct comparability with the
*Journal of International Economics* baseline. `RiskSurface.save` /
`RiskSurface.load` round-trip is exposed. `coord.fit()` takes
`crisis_label_col` as a kwarg and `coord.predict()` takes
`n_mc_samples` (default 30 for production, 1 for smoke tests).

**Mitigation.** `AuthorityGraph` exposes two classmethod factories:
`build_default(countries)` and `build_single_agent(countries)`; the
single-agent factory collapses all institutions to a single
`"central_planner"` per country owning every lever, operationalising
the E04 ablation that was deferred in v0.1.0. The `cbf` block exposes
`check_relative_degree(simulator)` for the relative-degree audit
reported in E05 of the paper; a `DifferentiableCBFLayer` is exposed for
end-to-end gradient training. `AsymmetricCost.reward` is keyword-only
(`reward(*, state, action, crisis_prob)`); `ReplayBuffer.push` takes
individual args rather than an `EpisodeStep` dataclass.

**Evaluation.** `WelfareEpisodeResult` now carries six Abadie-style
diagnostics: `pre_rmspe`, `post_rmspe`, `rmspe_ratio`,
`convex_hull_violation_pct`, `placebo_p_value`, `placebo_n_donors`. The
v0.2.0 `CounterfactualWelfare._placebo_test` implements Abadie (2010)
permutation inference: it fits SCM weights for each donor-as-placebo,
computes each placebo's RMSPE ratio, and reports the share that exceeds
the actual unit's ratio. `compare_models` accepts `metric="squared_error"`
or `"log_loss"` (v0.1.0 used the invalid `"brier"` string).
`MitigationAblationRunner` covers the three mitigation ablations
(`single_agent`, `no_cbf`, `symmetric_cost`) as in-process evaluations.
`CoordinatorAblationRunner._no_calibration` is corrected (monkey-patches
calibrators to identity *after* training, so the ablation isolates the
calibration contribution rather than confounding it with detector-fit
differences). `BaselineFactory` defaults to `label_col="crisis_pre_3y"`
matching the v0.2.0 Bluwstein-faithful headline horizon.

**Training & reproducibility.** Phase D directory is `phase_d_jointtune`
(was `phase_d_joint`). The orchestrator now (i) detects whether
`PYTHONHASHSEED` is set at startup and warns loudly if not (Python
ignores runtime `os.environ` writes for hash randomisation), (ii)
records the value alongside the seed in `manifest.json` under a new
`reproducibility` field, (iii) calls `torch.use_deterministic_algorithms`
with `warn_only=False` by default — opt out with the new
`--no-strict-determinism` CLI flag, (iv) emits a structured
`components` field in each phase's diagnostics block, (v) warns on any
unknown top-level or mitigation key in the YAML config (`v0.1.0`
silently ignored typos like `mitigatian:` or `single_agent_mode_:`),
(vi) supports `mitigation.single_agent_mode` directly via the
orchestrator (no need to manually swap the authority graph).

**Experiments.** E02 now reports dual-horizon detection (2-year and
3-year). E04 is a full mitigation-ablation table using
`MitigationAblationRunner`, including the new single-agent row. E05
reports the relative-degree audit plus `last_projection_info` summary.
E06 LaTeX includes the new Abadie diagnostics columns. `KeyboardInterrupt`
is re-raised by `run_one_experiment` (v0.1.0 swallowed it via broad
`except Exception`). The `--data-checksum ''` empty-string mode prints
the actual panel checksum without verifying — handy for first-time users
who don't yet know the expected value.

**Configs.** New `ablation_single_agent.yaml` for the E04 single-agent
baseline. Every training config now has `coordinator.crisis_label_col`,
`coordinator.predict_n_mc_samples`, `mitigation.single_agent_mode`,
`rnn_varselect.use_glu_gating`, and `network_contagion.bilateral_adj_path`
keys. `data_pipeline.yaml` carries `filter_method` (`"hp"` default;
`"hamilton"` alternative). Cross-config consistency is asserted by
`test_config_drift.py`.

---

## Quick start

### 1. Install

The canonical reproduction environment is a conda environment based on
CentOS 7 with the `gxx_linux-64=11` toolchain workaround. Non-CentOS
users (Ubuntu, macOS) can drop that pin — see notes in `environment.yml`.

```bash
# Clone (URL blinded for peer review):
git clone <repo-url> em-fin-stability
cd em-fin-stability

# Create the conda environment (~10-15 min):
conda env create -f environment.yml
conda activate em-fin-stability

# Install the package in editable mode:
pip install -e .

# Optional dev / test extras:
pip install -e ".[dev]"

# Optional plotting / figure-rendering stack (NOT used by src/):
pip install -e ".[plots]"

# Optional live-download deps for rebuilding data/raw cache:
pip install -e ".[online]"
```

**Critical reproducibility step**: export `PYTHONHASHSEED` *before*
launching Python:

```bash
export PYTHONHASHSEED=42
```

Python's hash randomisation is decided at interpreter startup; runtime
`os.environ` writes have no effect. The v0.2.0 orchestrator warns loudly
if this is unset (and records the actual value in `manifest.json`), but
it cannot fix the situation retroactively.

### 2. Smoke test (~3 minutes)

Confirms the full pipeline runs end-to-end with toy hyper-parameters.
Does **not** produce any scientific claim — it just confirms imports,
shape contracts, and phase chaining all work.

```bash
emfs-train \
    --config configs/smoke_test.yaml \
    --panel data/processed/em_panel \
    --output results/smoke_test \
    --seed 42
```

If this fails on a fresh install, that's a configuration or environment
issue, not a code issue — please open a GitHub issue (link revealed at
acceptance) with the full error output and the output of `conda list`.

### 3. Single experiment (~10-30 minutes)

The seven paper experiments are E01-E07. For development iteration:

```bash
# E02: headline coordinator-vs-baselines detection result.
emfs-experiments \
    --panel data/processed/em_panel \
    --run results/smoke_test \
    --output results/e02 \
    --mode single --experiment E02 --seed 42

# See the full list:
emfs-experiments --list
```

### 4. Full paper replication (~22-28 hours on 4× RTX 3090)

```bash
export PYTHONHASHSEED=42
emfs-experiments \
    --panel data/processed/em_panel \
    --config configs/replicate.yaml \
    --output results/replicate \
    --mode replicate \
    --data-checksum "$(cat data/processed/em_panel/CHECKSUM)"
```

This runs all seven experiments across the five paper seeds
`{42, 1729, 2718, 3141, 8128}` and writes camera-ready LaTeX tables to
`results/replicate/E07_cross_seed/`.

Total wall-clock on the target hardware is ~22-28 hours
(replicate.yaml is roughly 5x longer per seed than default.yaml because
the coordinator runs 150 epochs vs 50 and Phase C runs 1000 MARL
episodes vs 200).

---

## Repository layout

```
em-fin-stability/
├── src/em_fin_stability/        # The seven core code modules:
│   ├── data_pipeline.py         #   multi-source EM panel + HamiltonFilter
│   ├── detection_agents.py      #   four detectors + adjacency providers
│   ├── coordinator.py           #   contribution #1: disagreement-aware coord
│   ├── mitigation_agents.py     #   contribution #2: authority-constrained MARL
│   ├── training.py              #   four-phase training orchestrator
│   ├── evaluation.py            #   metrics, baselines, Abadie placebo
│   └── experiments.py           #   CLI dispatch for E01-E07
├── configs/                     # YAML training configurations:
│   ├── default.yaml             #   canonical paper config
│   ├── replicate.yaml           #   longer training, 5-seed paper-grade
│   ├── smoke_test.yaml          #   fast end-to-end verification
│   ├── data_pipeline.yaml       #   data-build config
│   ├── ablation_no_disagreement.yaml
│   ├── ablation_no_cbf.yaml
│   ├── ablation_symmetric_cost.yaml
│   └── ablation_single_agent.yaml      # new in v0.2.0 (E04 baseline)
├── data/                        # Data root:
│   ├── raw/                     #   source-specific downloads (cached)
│   ├── interim/                 #   intermediate harmonised tables
│   └── processed/em_panel/      #   ProcessedPanel ready for training
├── results/                     # Run output directories (gitignored).
├── tests/                       # pytest suites (one per module + drift checks).
├── scripts/                     # Reproducibility runner shell scripts.
├── paper/                       # LaTeX source for the manuscript.
├── pyproject.toml               # PEP 621 metadata + tool configs.
├── environment.yml              # Conda environment.
├── CITATION.cff                 # Citation metadata (CFF 1.2.0).
├── LICENSE                      # MIT.
└── README.md                    # You are here.
```

### The seven core modules

| Module | Lines | Purpose |
|---|---:|---|
| `data_pipeline.py` | 2489 | Multi-source EM panel (28 economies), HP + Hamilton trend filters, `ProcessedPanel.checksum()` |
| `detection_agents.py` | 2204 | Four detectors with uniform `Detector` ABC, `DetectorOutput` dataclass, `BilateralAdjacencyProvider` ABC, `BISBilateralProvider` |
| `coordinator.py` | 1591 | Calibrator + 32-dim disagreement featurizer + router + aggregator + MC-Dropout (contribution #1) |
| `mitigation_agents.py` | 2005 | Authority graph (`build_default` / `build_single_agent`) + simulator + asymmetric cost + CBF + TD3 MARL (contribution #2) |
| `training.py` | 1499 | Four-phase orchestrator with `--resume`, PYTHONHASHSEED audit, strict-determinism toggle |
| `evaluation.py` | 1731 | Bluwstein baselines + DM/Wilcoxon + Abadie placebo + ablation runners (contribution #3) |
| `experiments.py` | 1285 | CLI dispatch for E01-E07 + LaTeX emission + cross-seed aggregation |

Total: ~12,800 lines of typed Python (~70% docstring-and-comment, ~30% code).

---

## Reproducibility

Every paper run produces a `manifest.json` at the run output root with:

- A UUID-based run ID and ISO-8601 UTC timestamp.
- Git commit hash + dirty flag (uncommitted local changes).
- Hostname, platform string, Python version, CPU and CUDA device info.
- Full `pip freeze` output sorted alphabetically.
- The complete config dict (post-validation).
- Master seed and per-phase derived seeds.
- A `reproducibility` block (new in v0.2.0) recording:
    - The value of `PYTHONHASHSEED` at startup (or `null` if unset).
    - Whether the value matched the master seed.
    - Whether `torch.use_deterministic_algorithms` was set with
      `warn_only=False`.
    - The actual cuDNN deterministic / benchmark flags.
- Per-phase status, wall-clock duration, checkpoint path, headline
  metrics, and a `components` structured-diagnostics dict.

The manifest is the single source-of-truth artifact for the
experimental setup.

### Per-seed determinism

The pipeline derives every RNG state from a single master seed via
`_seed_all` in `training.py`:

```python
def _seed_all(seed: int, *, strict_determinism: bool = True) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if strict_determinism:
        torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "seed": seed,
        "strict_determinism_requested": strict_determinism,
        "deterministic_algorithms_set": strict_determinism,
        "cudnn_deterministic": torch.backends.cudnn.deterministic
                                if torch.cuda.is_available() else None,
    }
```

`strict_determinism=True` is the v0.2.0 default. To allow non-deterministic
ops (faster on some hardware, useful for hyperparameter sweeps where
exact reproducibility doesn't matter), pass `--no-strict-determinism` to
the CLI.

Within one run, every per-component RNG is derived from the master seed
via `_derived_seed(master, phase_label)`, which uses a fixed integer
offset rather than `hash(label)` — that's per-interpreter unstable
under `PYTHONHASHSEED` so a `hash`-based derivation would produce
different per-phase seeds across runs even with the same master.

In `replicate` mode the seed-loop re-calls `_seed_all` at the start of
each iteration so global numpy / random state doesn't drift between
seeds. Without this, only the first seed in the loop would be
bit-reproducible against a standalone single-seed run.

### Data checksums (v0.2.0)

`--data-checksum` on `emfs-experiments` asserts the input panel's
SHA-256 matches a recorded value. v0.2.0 computes this checksum via
`ProcessedPanel.checksum()` — a hash of the panel's *content* (features,
labels, splits) rather than the filesystem bytes (which v0.1.0 used
and which was sensitive to Parquet metadata and filesystem layout).

```bash
# Verify against a recorded checksum:
emfs-experiments \
    --panel data/processed/em_panel \
    --data-checksum "$(cat data/processed/em_panel/CHECKSUM)" \
    --config configs/replicate.yaml \
    --output results/replicate \
    --mode replicate

# Print the actual checksum without verifying (first-time setup):
emfs-experiments \
    --panel data/processed/em_panel \
    --data-checksum '' \
    --output results/dummy --list
```

---

## Data sources

The pipeline harmonises eight sources into a single quarterly EM panel
spanning 1995-2024 across 28 economies (24 MSCI EM + 4 non-MSCI study
countries). All sources require local cache files (delivered with the
repository; see `data/raw/README.md` for the redistribution-friendly
subset).

| Source | Variables | Frequency |
|---|---|---|
| IMF IFS | Output, inflation, exchange rates | Quarterly |
| IMF FSI | Bank-sector NPL, capital ratios | Quarterly |
| World Bank GFDD | Credit-to-GDP, bank concentration | Annual → quarterly |
| BIS LBS | External liabilities, bilateral exposures (v0.2.0) | Quarterly |
| Yahoo Finance | EM equity indices, volatility | Daily → quarterly |
| FRED EMBI | Sovereign-spread proxy | Daily → quarterly |
| GDELT | Sentiment events (optional) | Daily → quarterly |
| Laeven-Valencia | Banking-crisis dates and types | Annual |

The Laeven-Valencia (2020) catalogue covers 1970-2017. The five
post-2017 crisis episodes (Türkiye 2018, Argentina 2018-19, Sri Lanka
2022, Lebanon 2019, Ghana 2022) are manually extended via the
`POST_2017_CRISES` constant in `data_pipeline.py`, with source
documentation in the constant's docstring. Crisis dates are
cross-validated against the Bordo-Eichengreen-Klingebiel-Martínez-Peria
financial-crisis indicators.

### Country roster (v0.2.0)

**24 MSCI Emerging Markets** (constituents as of January 2025, after
Kuwait + UAE reclassifications): BRA, CHL, COL, MEX, PER (Americas);
CZE, EGY, GRC, HUN, KWT, POL, QAT, SAU, ZAF, TUR, ARE (EMEA); CHN, IND,
IDN, KOR, MYS, PHL, TWN, THA (Asia).

**4 non-MSCI study countries** (the post-2017 crisis subjects that
aren't in the current MSCI EM index): ARG, LBN, LKA, GHA.

These are exposed as `MSCI_EM_COUNTRIES` and
`NON_MSCI_STUDY_COUNTRIES` dicts in `data_pipeline.py`; the
backward-compatible `EM_COUNTRIES` alias is the 28-country union.

---

## Hardware requirements

| Workload | Minimum | Recommended | Notes |
|---|---|---|---|
| Smoke test | 4 GB RAM, CPU | 1× GPU | Runs on a laptop. |
| Single experiment | 16 GB RAM, 1× GPU (8 GB) | 1× RTX 3090 | E02-E03 are GPU-bound. |
| Full replicate (5 seeds × 7 experiments) | 64 GB RAM, 4× GPU (24 GB) | 4× RTX 3090 | ~22-28 hours. |
| Counterfactual welfare bootstrap (E06) | 32 GB RAM, 1× GPU | 4× GPU | Trivially parallelisable. |

The training orchestrator does **not** automatically distribute across
GPUs — Phase A's four detectors are fit sequentially. To use all four
RTX 3090s in parallel, set `CUDA_VISIBLE_DEVICES=0`, `=1`, etc. across
four shells launching different seeds simultaneously.

---

## Testing

```bash
# Full test suite (requires `pip install -e ".[dev]"`):
export PYTHONHASHSEED=42
pytest

# With coverage:
pytest --cov=em_fin_stability --cov-report=html

# Run only the config-drift check:
pytest tests/test_config_drift.py -v

# Run only the default (non-integration) tier — no torch/sklearn needed:
pytest -m "not integration"

# Run integration tier (needs torch + sklearn + scipy + statsmodels + cvxpy):
pytest -m integration
```

The default tier runs in under a minute on a laptop; the integration
tier needs the full conda environment and a GPU for the CBF /
MARL-policy tests.

The v0.2.0 test suite is split as: `test_config_drift.py` (cross-config
consistency, 28-country count, v0.2.0 schema), `test_data_pipeline.py`
(MSCI / non-MSCI split, 5-tuple `POST_2017_CRISES`, `HamiltonFilter`,
`ProcessedPanel.checksum`), `test_detection_agents.py` (registry, default
kwargs, adjacency providers), `test_coordinator.py` (32-feature
disagreement, MC-Dropout signature, `RiskSurface` save/load),
`test_mitigation_agents.py` (`build_default` / `build_single_agent`,
keyword-only `reward`, push args), `test_evaluation.py` (Abadie
diagnostics, `compare_models` metrics, `MitigationAblationResult`),
`test_training.py` (PYTHONHASHSEED check, strict-determinism, manifest
v0.2.0 schema), `test_experiments.py` (checksum via `ProcessedPanel`,
`KeyboardInterrupt` propagation, empty-checksum CLI mode).

---

## Citation

If you use this software, please cite both the paper (preferred
citation in `CITATION.cff`) and the software itself:

```bibtex
@article{REDACTED2026,
  title   = {From Risk Detection to Risk Mitigation:
             A Multi-Agent Approach for Dynamic Financial Stability
             in Emerging Markets},
  author  = {{Anonymous Authors}},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Under review; identifying details withheld for peer review.}
}

@software{REDACTED_software_v020,
  title   = {em\_fin\_stability v0.2.0:
             Reference implementation of an authority-constrained
             multi-agent crisis-mitigation framework},
  author  = {{Anonymous Authors}},
  year    = {2026},
  version = {0.2.0},
  note    = {Under review; identifying details withheld for peer review.}
}
```

The author and DOI fields will be populated at acceptance. GitHub
displays the CFF metadata directly when you click *"Cite this
repository"*.

---

## License

MIT. See `LICENSE`.

---

## Acknowledgments

Listed at acceptance, once peer review is complete. The codebase builds
directly on prior open-source work by Kraevskiy, Prokhorov &
Sokolovskiy (2024); Aquilina, Araujo, Gelos, Park & Pérez-Cruz (2025);
Li, Tam & Yeung (2024); Bluwstein, Buckmann, Joseph, Kang, Kapadia &
Şimşek (2023); Hamilton (2018); and Abadie, Diamond & Hainmueller
(2010). See `CITATION.cff` for the full reference list with citations
to the BIS LBS bilateral dataset and the Laeven-Valencia (2020) crisis
catalogue.
