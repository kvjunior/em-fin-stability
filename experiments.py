"""
experiments.py
==============

Ablation-study orchestrator.

The framework's central scientific claims — that retrieval, case-
coherence regularisation, the coordinator, and the control-barrier
function each contribute meaningfully — require an ablation table
that runs the framework with each component disabled in turn,
measures the resulting metrics, and compares against the full
framework. This module produces that table.

Architecture
------------
::

    ExperimentConfig
        ├── ablations: list[AblationSpec]
        │       ├── name: "full"
        │       ├── name: "no_retrieval"
        │       ├── name: "no_case_coherence"
        │       ├── name: "no_coordinator"
        │       └── name: "no_cbf"
        ├── seeds: list[int]  (e.g. [42, 43, 44, 45, 46])
        ├── base_training_config: TrainingConfig
        ├── eval_kwargs: dict (mitigation_n_episodes, episode_len, ...)
        └── output_dir: Path

    Experiment(config, library_factory)
        │
        ├── run_cell(ablation, seed) → CellResult
        │   ├── apply ablation overrides to TrainingConfig
        │   ├── TrainingOrchestrator.run_all_phases()
        │   └── Evaluator.evaluate_all() with ablation flags
        │
        ├── run_all() → ExperimentResults
        │   (sequentially runs every (ablation × seed) cell, with
        │   resumability: cells with existing manifest are loaded)
        │
        └── ExperimentResults
            ├── per_cell: dict[(ablation, seed), CellResult]
            ├── aggregate() → table of (ablation × metric) mean ± std
            ├── significance_test(metric, baseline="full")
            │   → Bonferroni-corrected paired permutation tests
            ├── to_markdown_table(metrics)
            └── to_latex_table(metrics) ← ready for \\input{} in paper

Ablation semantics
------------------
* **full**: default TrainingConfig and default eval flags.
* **no_retrieval**: ``mitigation_k_retrieval = 1`` (top-1 only) and
  ``mitigation_case_coherence_weight = 0.0`` (no analogy gradient
  shapes the actor). The analogy engine is still trained and the
  retrieval is still computed; it just has no influence on the
  policy. This isolates retrieval-augmentation as a learning signal.
* **no_case_coherence**: ``mitigation_case_coherence_weight = 0.0``.
  Retrieval still happens at inference (the context vector is still
  consumed by the actor), but the coherence regulariser is removed
  during training.
* **no_coordinator**: ``mitigation_use_ground_truth_type = False``.
  At training time the policy sees a uniform type prior rather than
  the ground-truth crisis label, simulating what happens when the
  coordinator is unavailable. The coordinator is still trained and
  evaluated normally (its metrics stay in the table).
* **no_cbf**: ``apply_cbf = False`` is passed through to
  ``Evaluator.evaluate_mitigation`` and
  ``Evaluator.evaluate_synthetic_control``. The CBF is still built
  during training (the policy network has the same parameter
  count), but it is not applied at evaluation time. This isolates
  the safety contribution of the CBF projection.

Reproducibility commitments
---------------------------
* Each (ablation, seed) cell has a deterministically-derived seed
  passed through to all underlying training and evaluation phases.
* Cell-level checkpointing: a cell completes when its
  ``cell_manifest.json`` is written; subsequent runs load it instead
  of re-training. To force a fresh run, delete the cell's
  subdirectory.
* The top-level experiment manifest records the
  ``base_config.fingerprint()`` and per-cell config overrides, so
  the exact experiment configuration is reconstructible from disk.

Significance testing
--------------------
For each metric and each non-baseline ablation A:
  * H0: mean(metric | A) == mean(metric | full)
  * Test: paired permutation across the matched-seed pairs
  * Statistic: difference in means
  * p-value: fraction of permuted sign assignments whose absolute
    difference equals or exceeds the observed one
Bonferroni correction multiplies p-values by the number of
ablations being compared (so 4 comparisons × p_raw).

References (APA-7)
------------------
Ernst, M. D. (2004). Permutation methods: A basis for exact
    inference. Statistical Science, 19(4), 676-685.

Holm, S. (1979). A simple sequentially rejective multiple test
    procedure. Scandinavian Journal of Statistics, 6(2), 65-70.

Version
-------
1.0.0  Camera-ready KBS submission.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Union

import numpy as np

from case_memory import CaseLibrary, CaseMemoryError
from evaluation import (
    EvaluationError,
    Evaluator,
)
from training import (
    TrainingConfig,
    TrainingError,
    TrainingOrchestrator,
)


__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "DEFAULT_SEEDS",
    "MAIN_TABLE_METRICS",
    "AVAILABLE_ABLATIONS",
    # Exceptions
    "ExperimentError",
    "CellFailedError",
    # Dataclasses
    "AblationSpec",
    "CellResult",
    "ExperimentConfig",
    "ExperimentResults",
    # Runner
    "Experiment",
    # Convenience
    "run_main_table",
    "paired_permutation_test",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Semantic version of the on-disk experiment manifest.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: Default seeds for the main ablation table. Five seeds give
#: sufficient variance estimation for paired permutation tests while
#: keeping runtime manageable (5 seeds × 5 ablations = 25 cells).
DEFAULT_SEEDS: Final[tuple[int, ...]] = (42, 43, 44, 45, 46)

#: Metric names that appear in the main results table. Each is a
#: dotted path into the evaluation manifest's per-family dicts.
MAIN_TABLE_METRICS: Final[tuple[str, ...]] = (
    "detection.macro_auc_pr",
    "coordinator.top_1_accuracy",
    "coordinator.macro_f1",
    "retrieval.recall_at_k.1",
    "retrieval.mrr",
    "mitigation.mean_episode_reward",
    "mitigation.mean_safety_violation_rate",
    "mitigation.mean_case_coherence",
    "mitigation.mean_output_loss_reduction_pct",
    "synthetic_control.mean_treatment_effect",
)

#: Available ablation names. Used for validation in ExperimentConfig.
AVAILABLE_ABLATIONS: Final[tuple[str, ...]] = (
    "full",
    "no_retrieval",
    "no_case_coherence",
    "no_coordinator",
    "no_cbf",
)

#: Default number of permutations for significance testing.
DEFAULT_N_PERMUTATIONS: Final[int] = 10_000

#: Default confidence level for bootstrap CIs in aggregation.
DEFAULT_CONFIDENCE: Final[float] = 0.95


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ExperimentError(CaseMemoryError):
    """Base class for experiments-module exceptions."""


class CellFailedError(ExperimentError):
    """A specific (ablation, seed) cell failed and could not produce
    a CellResult."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_hex(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sanitise_for_json(obj: Any) -> Any:
    """Replace non-finite floats with None for JSON serialisation."""
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitise_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise_for_json(v) for v in obj]
    return obj


def _extract_metric_path(
    manifest: dict[str, Any], dotted_path: str,
) -> Optional[float]:
    """Pull a metric value out of a nested manifest by dotted path.

    Returns None if any segment is missing. Numeric coercion:
    returns float if the leaf is a number, else None.

    Special handling: paths into ``recall_at_k.<int>`` look up by
    the string key (since JSON dict keys are strings).
    """
    parts = dotted_path.split(".")
    cur: Any = manifest
    for p in parts:
        if isinstance(cur, dict):
            if p in cur:
                cur = cur[p]
            else:
                return None
        else:
            return None
    if isinstance(cur, (int, float, np.integer, np.floating)) and not isinstance(cur, bool):
        v = float(cur)
        return v if math.isfinite(v) else None
    return None


# ---------------------------------------------------------------------------
# AblationSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationSpec:
    """Specification of a single ablation cell.

    An ablation modifies the base ``TrainingConfig`` via
    ``training_overrides`` (a dict of field names to override values)
    and the evaluation call via ``eval_overrides`` (a dict passed as
    kwargs to ``Evaluator.evaluate_all``).

    The five canonical ablations are constructed by the
    ``AblationSpec.from_name`` factory.
    """
    name: str
    training_overrides: dict[str, Any] = field(default_factory=dict)
    eval_overrides: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_name(cls, name: str) -> "AblationSpec":
        """Build a canonical AblationSpec by name."""
        if name == "full":
            return cls(
                name="full",
                description="Full framework with all components enabled.",
            )
        if name == "no_retrieval":
            return cls(
                name="no_retrieval",
                training_overrides={
                    "mitigation_k_retrieval": 1,
                    "mitigation_case_coherence_weight": 0.0,
                },
                description=(
                    "Retrieval shrunk to top-1 and case-coherence "
                    "regulariser removed. Tests whether retrieval-"
                    "augmentation contributes to mitigation quality."
                ),
            )
        if name == "no_case_coherence":
            return cls(
                name="no_case_coherence",
                training_overrides={
                    "mitigation_case_coherence_weight": 0.0,
                },
                description=(
                    "Case-coherence regulariser removed during training. "
                    "Retrieval still happens but exerts no gradient "
                    "pressure on the actor."
                ),
            )
        if name == "no_coordinator":
            return cls(
                name="no_coordinator",
                training_overrides={
                    "mitigation_use_ground_truth_type": False,
                },
                description=(
                    "Uniform type prior at training time. Tests "
                    "whether the coordinator's crisis-type signal "
                    "helps the policy."
                ),
            )
        if name == "no_cbf":
            return cls(
                name="no_cbf",
                eval_overrides={"apply_cbf": False},
                description=(
                    "Control-barrier-function safety projection "
                    "disabled at evaluation time. Tests whether CBF "
                    "improves safety-violation rate and reward."
                ),
            )
        raise ExperimentError(
            f"AblationSpec.from_name: unknown ablation {name!r}; "
            f"available: {list(AVAILABLE_ABLATIONS)}"
        )

    def apply_to_config(
        self, base: TrainingConfig, *, output_dir: Path,
    ) -> TrainingConfig:
        """Build a TrainingConfig for this ablation.

        Validates that every key in ``training_overrides`` is a real
        TrainingConfig field; unknown keys raise.
        """
        cfg_dict = dataclasses.asdict(base)
        cfg_dict["output_dir"] = Path(output_dir)
        valid_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
        for k, v in self.training_overrides.items():
            if k not in valid_fields:
                raise ExperimentError(
                    f"AblationSpec[{self.name}]: training_overrides "
                    f"contains unknown field {k!r}; valid: "
                    f"{sorted(valid_fields)}"
                )
            cfg_dict[k] = v
        return TrainingConfig(**cfg_dict)


# ---------------------------------------------------------------------------
# CellResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    """Result of a single (ablation, seed) cell."""
    ablation_name: str
    seed: int
    status: str  # "complete" | "loaded" | "failed"
    started_at: str
    finished_at: str
    duration_seconds: float
    artefact_dir: str
    training_config_fingerprint: str
    evaluation_manifest: dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return _sanitise_for_json({
            "ablation_name": self.ablation_name,
            "seed": int(self.seed),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": float(self.duration_seconds),
            "artefact_dir": str(self.artefact_dir),
            "training_config_fingerprint": self.training_config_fingerprint,
            "evaluation_manifest": self.evaluation_manifest,
            "error": self.error,
        })

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CellResult":
        return cls(
            ablation_name=str(d["ablation_name"]),
            seed=int(d["seed"]),
            status=str(d["status"]),
            started_at=str(d["started_at"]),
            finished_at=str(d["finished_at"]),
            duration_seconds=float(d["duration_seconds"]),
            artefact_dir=str(d["artefact_dir"]),
            training_config_fingerprint=str(d["training_config_fingerprint"]),
            evaluation_manifest=dict(d.get("evaluation_manifest") or {}),
            error=(str(d["error"]) if d.get("error") else None),
        )

    def get_metric(self, dotted_path: str) -> Optional[float]:
        """Extract a metric from this cell's evaluation manifest."""
        return _extract_metric_path(self.evaluation_manifest, dotted_path)


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Top-level experiment specification.

    Parameters
    ----------
    ablations : list[AblationSpec]
        Cells to run. Must include exactly one with ``name="full"``
        if significance testing is desired (it serves as the
        baseline).
    seeds : tuple[int, ...]
        Seeds to use for each ablation.
    base_training_config : TrainingConfig
        Default training configuration. Each cell's TrainingConfig
        is this with the ablation's overrides applied.
    library_factory : Callable[[int], CaseLibrary]
        Function ``seed -> CaseLibrary`` used to build the library
        for each seed. The same seed produces the same library
        (for reproducibility). For the canonical synthetic-library
        experiments, this would be
        ``lambda s: build_synthetic_library(seed=s, n_per_type=5)``.
    output_dir : Path
        Root directory for all cells' artefacts.
    eval_kwargs : dict
        Keyword arguments forwarded to ``Evaluator.evaluate_all``
        (e.g. ``mitigation_n_episodes``, ``n_bootstrap``).
    device : str
        Forwarded to TrainingConfig and Evaluator.
    verbose : bool
    """
    ablations: list[AblationSpec]
    seeds: tuple[int, ...]
    base_training_config: TrainingConfig
    library_factory: Callable[[int], CaseLibrary]
    output_dir: Path
    eval_kwargs: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.ablations:
            raise ExperimentError(
                "ExperimentConfig: at least one ablation required"
            )
        names = [a.name for a in self.ablations]
        if len(set(names)) != len(names):
            raise ExperimentError(
                f"ExperimentConfig: duplicate ablation names: {names}"
            )
        if not self.seeds:
            raise ExperimentError("ExperimentConfig: at least one seed required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ExperimentError(
                f"ExperimentConfig: duplicate seeds: {self.seeds}"
            )
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        if not callable(self.library_factory):
            raise ExperimentError(
                "ExperimentConfig: library_factory must be callable"
            )

    @property
    def n_cells(self) -> int:
        return len(self.ablations) * len(self.seeds)

    @property
    def ablation_names(self) -> list[str]:
        return [a.name for a in self.ablations]

    def fingerprint(self) -> str:
        """SHA-256 short digest of the experiment config (excluding
        the library_factory which isn't serialisable).
        """
        payload = {
            "ablations": [
                {
                    "name": a.name,
                    "training_overrides": a.training_overrides,
                    "eval_overrides": a.eval_overrides,
                }
                for a in self.ablations
            ],
            "seeds": list(self.seeds),
            "base_config_fingerprint": self.base_training_config.fingerprint(),
            "eval_kwargs": self.eval_kwargs,
            "device": self.device,
        }
        return _sha256_hex(_canonical_json(payload))[:16]


# ---------------------------------------------------------------------------
# ExperimentResults
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResults:
    """All cell results for an experiment, plus aggregation methods.

    The primary interface is ``aggregate()`` which produces a
    table of (ablation × metric) → (mean, std, ci_lo, ci_hi).
    """
    config_fingerprint: str
    library_size: int
    ablation_names: list[str]
    seeds: list[int]
    cells: dict[str, CellResult]  # key: f"{ablation}_seed{seed}"

    def get_cell(
        self, ablation: str, seed: int,
    ) -> Optional[CellResult]:
        return self.cells.get(self._cell_key(ablation, seed))

    @staticmethod
    def _cell_key(ablation: str, seed: int) -> str:
        return f"{ablation}_seed{seed}"

    def values_for(
        self, ablation: str, metric: str,
    ) -> np.ndarray:
        """Per-seed metric values for an ablation.

        Returns an array of shape ``(n_seeds,)`` with NaN entries for
        any cell that failed or whose metric is missing.
        """
        vals = []
        for s in self.seeds:
            cell = self.get_cell(ablation, s)
            if cell is None:
                vals.append(float("nan"))
                continue
            v = cell.get_metric(metric)
            vals.append(float(v) if v is not None else float("nan"))
        return np.asarray(vals, dtype=np.float64)

    def aggregate(
        self,
        metrics: Optional[Sequence[str]] = None,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Mean ± std per (ablation × metric).

        Returns
        -------
        table : dict[ablation_name -> dict[metric -> dict[stat -> float]]]
            ``stat`` keys: ``"mean"``, ``"std"``, ``"n"``,
            ``"sem"`` (standard error of the mean).
        """
        if metrics is None:
            metrics = MAIN_TABLE_METRICS
        out: dict[str, dict[str, dict[str, float]]] = {}
        for a in self.ablation_names:
            out[a] = {}
            for m in metrics:
                vals = self.values_for(a, m)
                finite = vals[np.isfinite(vals)]
                n = int(finite.size)
                if n == 0:
                    out[a][m] = {
                        "mean": float("nan"),
                        "std": float("nan"),
                        "n": 0,
                        "sem": float("nan"),
                    }
                else:
                    out[a][m] = {
                        "mean": float(np.mean(finite)),
                        "std": float(np.std(finite, ddof=1) if n > 1 else 0.0),
                        "n": n,
                        "sem": float(
                            np.std(finite, ddof=1) / np.sqrt(n)
                            if n > 1 else 0.0
                        ),
                    }
        return out

    def significance_tests(
        self,
        metrics: Optional[Sequence[str]] = None,
        *,
        baseline: str = "full",
        n_permutations: int = DEFAULT_N_PERMUTATIONS,
        seed: int = 0,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Paired permutation tests vs ``baseline``.

        For each ablation A != baseline and each metric M:
          * Form matched pairs (M_A[s], M_baseline[s]) over seeds s.
          * Test H0: mean difference == 0.
          * Compute Bonferroni-corrected p-value.

        Returns
        -------
        sig : dict[ablation_name -> dict[metric -> {"diff": ..., "p_raw": ...,
                                                   "p_bonferroni": ...}]]
        """
        if baseline not in self.ablation_names:
            raise ExperimentError(
                f"significance_tests: baseline {baseline!r} not in "
                f"ablation_names {self.ablation_names}"
            )
        if metrics is None:
            metrics = MAIN_TABLE_METRICS
        non_baseline = [a for a in self.ablation_names if a != baseline]
        n_comparisons = max(1, len(non_baseline))
        out: dict[str, dict[str, dict[str, float]]] = {}
        for a in non_baseline:
            out[a] = {}
            for m in metrics:
                vals_a = self.values_for(a, m)
                vals_b = self.values_for(baseline, m)
                mask = np.isfinite(vals_a) & np.isfinite(vals_b)
                if mask.sum() < 2:
                    out[a][m] = {
                        "diff": float("nan"),
                        "p_raw": float("nan"),
                        "p_bonferroni": float("nan"),
                        "n_paired": int(mask.sum()),
                    }
                    continue
                paired_a = vals_a[mask]
                paired_b = vals_b[mask]
                diff, p_raw = paired_permutation_test(
                    paired_a, paired_b,
                    n_permutations=n_permutations,
                    seed=seed,
                )
                p_bonferroni = min(1.0, p_raw * n_comparisons)
                out[a][m] = {
                    "diff": float(diff),
                    "p_raw": float(p_raw),
                    "p_bonferroni": float(p_bonferroni),
                    "n_paired": int(mask.sum()),
                }
        return out

    def to_markdown_table(
        self,
        metrics: Optional[Sequence[str]] = None,
        *,
        decimals: int = 3,
    ) -> str:
        """Render a markdown table of mean ± std per (ablation × metric)."""
        if metrics is None:
            metrics = MAIN_TABLE_METRICS
        agg = self.aggregate(metrics)
        cols = ["ablation"] + list(metrics)
        rows: list[list[str]] = [cols]
        for a in self.ablation_names:
            row = [a]
            for m in metrics:
                stats = agg[a][m]
                if math.isnan(stats["mean"]):
                    row.append("nan")
                else:
                    row.append(
                        f"{stats['mean']:.{decimals}f} ± "
                        f"{stats['std']:.{decimals}f}"
                    )
            rows.append(row)
        # Markdown formatting
        header = "| " + " | ".join(rows[0]) + " |"
        separator = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = "\n".join(
            "| " + " | ".join(r) + " |" for r in rows[1:]
        )
        return f"{header}\n{separator}\n{body}"

    def to_latex_table(
        self,
        metrics: Optional[Sequence[str]] = None,
        *,
        decimals: int = 3,
        caption: str = "Main ablation results.",
        label: str = "tab:main_results",
    ) -> str:
        """Render a LaTeX table ready for ``\\input{}``.

        Uses ``\\booktabs``-style rules (``\\toprule``, ``\\midrule``,
        ``\\bottomrule``); requires ``\\usepackage{booktabs}`` in the
        target document.
        """
        if metrics is None:
            metrics = MAIN_TABLE_METRICS
        agg = self.aggregate(metrics)
        n_cols = 1 + len(metrics)
        col_spec = "l" + "c" * len(metrics)
        # Use short metric names in the header (last dotted segment)
        short_names = [m.split(".")[-1] for m in metrics]
        # Escape LaTeX-sensitive chars
        def esc(s: str) -> str:
            return s.replace("_", r"\_")
        header_cells = ["Ablation"] + [esc(s) for s in short_names]
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{" + caption + r"}",
            r"\label{" + label + r"}",
            r"\begin{tabular}{" + col_spec + r"}",
            r"\toprule",
            " & ".join(header_cells) + r" \\",
            r"\midrule",
        ]
        for a in self.ablation_names:
            cells = [esc(a)]
            for m in metrics:
                stats = agg[a][m]
                if math.isnan(stats["mean"]):
                    cells.append("---")
                else:
                    cells.append(
                        f"{stats['mean']:.{decimals}f} $\\pm$ "
                        f"{stats['std']:.{decimals}f}"
                    )
            lines.append(" & ".join(cells) + r" \\")
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        # Ignore n_cols (we use col_spec); silence linter
        del n_cols
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        complete = sum(
            1 for c in self.cells.values() if c.status in ("complete", "loaded")
        )
        failed = sum(
            1 for c in self.cells.values() if c.status == "failed"
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": self.config_fingerprint,
            "library_size": int(self.library_size),
            "n_ablations": len(self.ablation_names),
            "n_seeds": len(self.seeds),
            "n_cells_total": len(self.ablation_names) * len(self.seeds),
            "n_cells_complete": int(complete),
            "n_cells_failed": int(failed),
            "ablations": list(self.ablation_names),
            "seeds": list(self.seeds),
        }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def paired_permutation_test(
    a: np.ndarray, b: np.ndarray, *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
    two_sided: bool = True,
) -> tuple[float, float]:
    """Paired permutation test on two arrays of equal length.

    Tests H0: mean(a - b) = 0 by randomly flipping the sign of
    individual paired differences.

    Parameters
    ----------
    a, b : np.ndarray of shape (N,)
        Matched samples (e.g., metric values from the same seeds
        under two ablations).
    n_permutations : int
    seed : int
    two_sided : bool
        If True, compute the two-sided p-value (fraction of
        permutations whose absolute mean equals or exceeds the
        absolute observed mean).

    Returns
    -------
    observed_diff : float
        Mean of (a - b).
    p_value : float
        In [0, 1]. The fraction of permutations producing an extreme
        statistic. With ``n_permutations`` permutations and a
        continuity correction (+1 in numerator and denominator), the
        minimum achievable p-value is ``1 / (n_permutations + 1)``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ExperimentError(
            f"paired_permutation_test: shape mismatch {a.shape} vs {b.shape}"
        )
    if a.ndim != 1:
        raise ExperimentError(
            f"paired_permutation_test: arrays must be 1-D; got {a.shape}"
        )
    if a.size < 2:
        raise ExperimentError(
            f"paired_permutation_test: need >= 2 paired samples; got {a.size}"
        )
    if n_permutations < 1:
        raise ExperimentError(
            f"paired_permutation_test: n_permutations must be >= 1; "
            f"got {n_permutations}"
        )
    diffs = a - b
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    # Permutation: random sign flips on each paired difference
    n = diffs.size
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    permuted_means = (signs * diffs).mean(axis=1)
    if two_sided:
        as_or_more_extreme = (
            np.abs(permuted_means) >= abs(observed) - 1e-12
        )
    else:
        # One-sided: count permutations >= observed (regardless of sign of observed)
        as_or_more_extreme = (
            permuted_means >= observed - 1e-12
            if observed >= 0 else
            permuted_means <= observed + 1e-12
        )
    # Add-one continuity correction (standard for exact permutation tests)
    p_value = (1 + int(as_or_more_extreme.sum())) / (n_permutations + 1)
    return observed, float(p_value)


# ---------------------------------------------------------------------------
# Experiment (the runner)
# ---------------------------------------------------------------------------


class Experiment:
    """Sequential cell-by-cell experiment runner with resumability.

    Parameters
    ----------
    config : ExperimentConfig
    """

    def __init__(self, config: ExperimentConfig) -> None:
        if not isinstance(config, ExperimentConfig):
            raise ExperimentError(
                f"Experiment: config must be ExperimentConfig; "
                f"got {type(config)}"
            )
        self.config: ExperimentConfig = config
        self.output_dir: Path = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cells: dict[str, CellResult] = {}

    # ------------------------------------------------------------------ #
    # Cell management
    # ------------------------------------------------------------------ #

    def _cell_dir(self, ablation_name: str, seed: int) -> Path:
        return self.output_dir / f"{ablation_name}_seed{seed}"

    def _cell_manifest_path(self, ablation_name: str, seed: int) -> Path:
        return self._cell_dir(ablation_name, seed) / "cell_manifest.json"

    def run_cell(
        self,
        ablation: AblationSpec,
        seed: int,
        *,
        force_retrain: bool = False,
    ) -> CellResult:
        """Run a single (ablation, seed) cell.

        Resumability: if the cell manifest already exists and
        ``force_retrain=False``, load it and return.
        """
        cell_key = ExperimentResults._cell_key(ablation.name, seed)
        cell_dir = self._cell_dir(ablation.name, seed)
        manifest_path = self._cell_manifest_path(ablation.name, seed)
        started_at = _utc_now_iso()
        t0 = time.time()

        # Resumability check
        if not force_retrain and manifest_path.is_file():
            try:
                cell_dict = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                cell = CellResult.from_dict(cell_dict)
                # Mark as loaded if it was previously complete
                if cell.status == "complete":
                    cell = CellResult(
                        ablation_name=cell.ablation_name,
                        seed=cell.seed,
                        status="loaded",
                        started_at=cell.started_at,
                        finished_at=cell.finished_at,
                        duration_seconds=cell.duration_seconds,
                        artefact_dir=cell.artefact_dir,
                        training_config_fingerprint=cell.training_config_fingerprint,
                        evaluation_manifest=cell.evaluation_manifest,
                        error=cell.error,
                    )
                self._cells[cell_key] = cell
                logger.info(
                    "Experiment: cell %s (status=%s, loaded from %s)",
                    cell_key, cell.status, manifest_path,
                )
                return cell
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning(
                    "Experiment: stale/corrupt cell manifest at %s "
                    "(%s); retraining",
                    manifest_path, exc,
                )

        # Build cell-specific training config
        cfg = ablation.apply_to_config(
            self.config.base_training_config,
            output_dir=cell_dir / "training",
        )
        # Override seed and device
        cfg_dict = dataclasses.asdict(cfg)
        cfg_dict["output_dir"] = cell_dir / "training"
        cfg_dict["seed"] = int(seed)
        cfg_dict["device"] = self.config.device
        cfg_dict["verbose"] = self.config.verbose
        cfg = TrainingConfig(**cfg_dict)

        logger.info(
            "Experiment: running cell %s "
            "(ablation=%s, seed=%d, fingerprint=%s)",
            cell_key, ablation.name, seed, cfg.fingerprint(),
        )

        # Build library
        try:
            library = self.config.library_factory(int(seed))
        except Exception as exc:
            logger.error(
                "Experiment: cell %s: library_factory failed: %s",
                cell_key, exc,
            )
            cell = CellResult(
                ablation_name=ablation.name,
                seed=int(seed),
                status="failed",
                started_at=started_at,
                finished_at=_utc_now_iso(),
                duration_seconds=time.time() - t0,
                artefact_dir=str(cell_dir),
                training_config_fingerprint=cfg.fingerprint(),
                evaluation_manifest={},
                error=f"library_factory: {type(exc).__name__}: {exc}",
            )
            self._cells[cell_key] = cell
            _atomic_write_text(
                manifest_path,
                json.dumps(cell.to_dict(), sort_keys=True, indent=2,
                           ensure_ascii=True, allow_nan=False),
            )
            return cell
        if not isinstance(library, CaseLibrary):
            raise CellFailedError(
                f"library_factory must return CaseLibrary; got {type(library)}"
            )

        # Train
        try:
            orchestrator = TrainingOrchestrator(cfg, library)
            artefacts = orchestrator.run_all_phases(
                force_retrain=force_retrain,
            )
            if not artefacts.is_complete:
                raise CellFailedError(
                    "Phases incomplete after run_all_phases"
                )
        except Exception as exc:
            logger.error(
                "Experiment: cell %s: training failed: %s",
                cell_key, exc,
            )
            cell = CellResult(
                ablation_name=ablation.name,
                seed=int(seed),
                status="failed",
                started_at=started_at,
                finished_at=_utc_now_iso(),
                duration_seconds=time.time() - t0,
                artefact_dir=str(cell_dir),
                training_config_fingerprint=cfg.fingerprint(),
                evaluation_manifest={},
                error=f"training: {type(exc).__name__}: {exc}",
            )
            self._cells[cell_key] = cell
            _atomic_write_text(
                manifest_path,
                json.dumps(cell.to_dict(), sort_keys=True, indent=2,
                           ensure_ascii=True, allow_nan=False),
            )
            return cell

        # Evaluate
        try:
            evaluator = Evaluator(
                artefacts, library,
                seed=int(seed) + 10_000,  # decorrelate eval seed from training
                device=self.config.device,
            )
            eval_kwargs = dict(self.config.eval_kwargs)
            # Apply ablation eval overrides on top
            eval_kwargs.update(ablation.eval_overrides)
            manifest = evaluator.evaluate_all(
                save_to=cell_dir / "evaluation_manifest.json",
                **eval_kwargs,
            )
        except Exception as exc:
            logger.error(
                "Experiment: cell %s: evaluation failed: %s",
                cell_key, exc,
            )
            cell = CellResult(
                ablation_name=ablation.name,
                seed=int(seed),
                status="failed",
                started_at=started_at,
                finished_at=_utc_now_iso(),
                duration_seconds=time.time() - t0,
                artefact_dir=str(cell_dir),
                training_config_fingerprint=cfg.fingerprint(),
                evaluation_manifest={},
                error=f"evaluation: {type(exc).__name__}: {exc}",
            )
            self._cells[cell_key] = cell
            _atomic_write_text(
                manifest_path,
                json.dumps(cell.to_dict(), sort_keys=True, indent=2,
                           ensure_ascii=True, allow_nan=False),
            )
            return cell

        # Persist successful cell
        cell = CellResult(
            ablation_name=ablation.name,
            seed=int(seed),
            status="complete",
            started_at=started_at,
            finished_at=_utc_now_iso(),
            duration_seconds=time.time() - t0,
            artefact_dir=str(cell_dir),
            training_config_fingerprint=cfg.fingerprint(),
            evaluation_manifest=manifest,
        )
        self._cells[cell_key] = cell
        _atomic_write_text(
            manifest_path,
            json.dumps(cell.to_dict(), sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        logger.info(
            "Experiment: cell %s complete (%.2fs)",
            cell_key, cell.duration_seconds,
        )
        return cell

    def run_all(
        self, *, force_retrain: bool = False,
    ) -> ExperimentResults:
        """Run every (ablation, seed) cell in the configured order."""
        logger.info(
            "Experiment: starting %d cells (%d ablations × %d seeds), "
            "output_dir=%s",
            self.config.n_cells, len(self.config.ablations),
            len(self.config.seeds), self.output_dir,
        )
        cell_keys: list[str] = []
        for ablation in self.config.ablations:
            for seed in self.config.seeds:
                key = ExperimentResults._cell_key(ablation.name, seed)
                cell_keys.append(key)
                try:
                    self.run_cell(ablation, seed, force_retrain=force_retrain)
                except Exception as exc:
                    logger.exception(
                        "Experiment: run_cell raised unexpectedly for %s: %s",
                        key, exc,
                    )
                self._save_top_level_manifest()

        # Build sample library to get library_size (only if at least
        # one cell produced metrics). The library_size attribute is
        # nice-to-have; if the factory is expensive we'd prefer not to
        # rebuild.
        first_complete = next(
            (c for c in self._cells.values()
             if c.evaluation_manifest.get("library_size") is not None),
            None,
        )
        library_size = (
            int(first_complete.evaluation_manifest["library_size"])
            if first_complete else 0
        )

        results = ExperimentResults(
            config_fingerprint=self.config.fingerprint(),
            library_size=library_size,
            ablation_names=self.config.ablation_names,
            seeds=list(self.config.seeds),
            cells=dict(self._cells),
        )
        self._save_top_level_manifest(results=results)
        n_complete = sum(
            1 for c in self._cells.values()
            if c.status in ("complete", "loaded")
        )
        logger.info(
            "Experiment: complete (%d/%d cells succeeded)",
            n_complete, self.config.n_cells,
        )
        return results

    def _save_top_level_manifest(
        self, *, results: Optional[ExperimentResults] = None,
    ) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": self.config.fingerprint(),
            "ablations": [
                {
                    "name": a.name,
                    "description": a.description,
                    "training_overrides": a.training_overrides,
                    "eval_overrides": a.eval_overrides,
                }
                for a in self.config.ablations
            ],
            "seeds": list(self.config.seeds),
            "device": self.config.device,
            "eval_kwargs": self.config.eval_kwargs,
            "base_config_fingerprint": (
                self.config.base_training_config.fingerprint()
            ),
            "cells": {
                k: c.to_dict() for k, c in sorted(self._cells.items())
            },
            "summary": (results.summary() if results else {}),
            "saved_at": _utc_now_iso(),
        }
        manifest = _sanitise_for_json(manifest)
        _atomic_write_text(
            self.output_dir / "experiment_manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )

    @classmethod
    def load_results(
        cls, output_dir: Path,
    ) -> ExperimentResults:
        """Reconstruct ExperimentResults from a previously-saved
        experiment_manifest.json, without re-running cells."""
        output_dir = Path(output_dir)
        manifest_path = output_dir / "experiment_manifest.json"
        if not manifest_path.is_file():
            raise ExperimentError(
                f"load_results: no experiment manifest at {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cells = {
            k: CellResult.from_dict(d)
            for k, d in (manifest.get("cells") or {}).items()
        }
        ablation_names = [a["name"] for a in manifest.get("ablations", [])]
        seeds = [int(s) for s in manifest.get("seeds", [])]
        library_size = int(
            manifest.get("summary", {}).get("library_size", 0)
        )
        return ExperimentResults(
            config_fingerprint=manifest.get("config_fingerprint", ""),
            library_size=library_size,
            ablation_names=ablation_names,
            seeds=seeds,
            cells=cells,
        )

    def __repr__(self) -> str:
        return (
            f"Experiment(n_cells={self.config.n_cells}, "
            f"output_dir={self.output_dir})"
        )


# ---------------------------------------------------------------------------
# Convenience: canonical main-table runner
# ---------------------------------------------------------------------------


# Import here to avoid a circular import at module-load time.
from collections.abc import Sequence  # noqa: E402


def run_main_table(
    output_dir: Path,
    *,
    library_factory: Optional[Callable[[int], CaseLibrary]] = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_per_type: int = 5,
    base_training_config: Optional[TrainingConfig] = None,
    eval_kwargs: Optional[dict[str, Any]] = None,
    device: str = "cpu",
    verbose: bool = False,
    quick: bool = False,
) -> ExperimentResults:
    """Run the canonical 5-ablation × n-seeds main table.

    Convenience wrapper that constructs the default
    ``ExperimentConfig`` with all five canonical ablations
    (``full``, ``no_retrieval``, ``no_case_coherence``,
    ``no_coordinator``, ``no_cbf``).

    Parameters
    ----------
    output_dir : Path
    library_factory : callable, optional
        Function ``seed -> CaseLibrary``. Defaults to
        ``build_synthetic_library(n_per_type=n_per_type, seed=seed)``.
    seeds : sequence of int
    n_per_type : int
        Used only if ``library_factory`` is not provided.
    base_training_config : TrainingConfig, optional
        Defaults to the camera-ready settings, modulated by ``quick``.
    eval_kwargs : dict, optional
        Defaults are ``mitigation_n_episodes=50, mitigation_episode_len=12,
        n_bootstrap=500`` (or smaller if ``quick``).
    device : str
    verbose : bool
    quick : bool
        If True, use small epoch counts, episode counts, and bootstrap
        replicates suitable for a smoke test (the full run can take
        hours on real data).
    """
    from data_pipeline import build_synthetic_library
    if library_factory is None:
        def library_factory(s: int) -> CaseLibrary:
            return build_synthetic_library(n_per_type=n_per_type, seed=s)
    if base_training_config is None:
        if quick:
            base_training_config = TrainingConfig(
                output_dir=Path(output_dir),
                device=device,
                verbose=verbose,
                detection_n_epochs=20, detection_batch_size=16,
                coordinator_n_epochs=20, coordinator_batch_size=8,
                signature_n_epochs=20,
                retriever_n_epochs=20,
                mitigation_n_episodes=8, mitigation_episode_len=4,
                mitigation_warmup_steps=10, mitigation_batch_size=8,
                mitigation_buffer_capacity=1000,
            )
        else:
            base_training_config = TrainingConfig(
                output_dir=Path(output_dir),
                device=device,
                verbose=verbose,
            )
    if eval_kwargs is None:
        eval_kwargs = (
            dict(mitigation_n_episodes=4, mitigation_episode_len=4,
                 n_bootstrap=50)
            if quick else
            dict(mitigation_n_episodes=50, mitigation_episode_len=12,
                 n_bootstrap=500)
        )
    ablations = [
        AblationSpec.from_name(name) for name in AVAILABLE_ABLATIONS
    ]
    config = ExperimentConfig(
        ablations=ablations,
        seeds=tuple(int(s) for s in seeds),
        base_training_config=base_training_config,
        library_factory=library_factory,
        output_dir=Path(output_dir),
        eval_kwargs=eval_kwargs,
        device=device,
        verbose=verbose,
    )
    return Experiment(config).run_all()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> int:
    """CLI: ``python experiments.py <command> [args]``.

    Commands
    --------
    main_table <output_dir> [--seeds 42 43 ...] [--n-per-type N] [--quick]
        Run the canonical 5-ablation main-results table.
    summary <experiment_manifest_path>
        Print an experiment manifest summary.
    tables <output_dir> [--metrics M1 M2 ...]
        Re-emit markdown and LaTeX tables from a completed experiment
        directory (without re-running cells).
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="python experiments.py",
        description="Run ablation experiments.",
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_main = sub.add_parser(
        "main_table", help="Run the canonical main-results table.",
    )
    p_main.add_argument("output_dir", type=Path)
    p_main.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p_main.add_argument("--n-per-type", type=int, default=5)
    p_main.add_argument("--device", default="cpu")
    p_main.add_argument("--quick", action="store_true")

    p_sum = sub.add_parser("summary", help="Print an experiment manifest.")
    p_sum.add_argument("manifest_path", type=Path)

    p_tab = sub.add_parser(
        "tables", help="Re-emit markdown and LaTeX tables.",
    )
    p_tab.add_argument("output_dir", type=Path)
    p_tab.add_argument("--metrics", nargs="*", default=None)
    p_tab.add_argument("--decimals", type=int, default=3)

    args = parser.parse_args()
    _configure_logging(args.log_level)

    try:
        if args.command == "main_table":
            results = run_main_table(
                args.output_dir,
                seeds=tuple(args.seeds),
                n_per_type=args.n_per_type,
                device=args.device,
                quick=args.quick,
            )
            print(json.dumps(results.summary(), indent=2, sort_keys=True))
            return 0
        elif args.command == "summary":
            manifest = json.loads(
                args.manifest_path.read_text(encoding="utf-8")
            )
            print(json.dumps(
                {k: v for k, v in manifest.items() if k != "cells"},
                indent=2, sort_keys=True,
            ))
            return 0
        elif args.command == "tables":
            results = Experiment.load_results(args.output_dir)
            metrics = args.metrics if args.metrics else None
            md = results.to_markdown_table(metrics, decimals=args.decimals)
            tex = results.to_latex_table(metrics, decimals=args.decimals)
            md_path = args.output_dir / "main_table.md"
            tex_path = args.output_dir / "main_table.tex"
            _atomic_write_text(md_path, md + "\n")
            _atomic_write_text(tex_path, tex + "\n")
            print(f"Wrote {md_path}")
            print(f"Wrote {tex_path}")
            print()
            print(md)
            return 0
        else:  # pragma: no cover
            parser.print_help()
            return 2
    except (CaseMemoryError, TrainingError,
            EvaluationError, ExperimentError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
