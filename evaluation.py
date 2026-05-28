"""
evaluation.py
=============

Metric collection module.

Given a completed ``TrainingArtifacts`` (or a directory of trained
phase artefacts) and a ``CaseLibrary`` to evaluate on, computes the
five metric families that appear in the manuscript's experimental
section:

  1. **Detection metrics**: per-detector AUC-PR (Davis & Goadrich
     2006), expected calibration error (Naeini, Cooper, & Hauskrecht
     2015), Brier score (Brier 1950). PR-curve is preferred over
     ROC for the class-imbalanced regime of crisis labels (Saito &
     Rehmsmeier 2015).
  2. **Coordinator metrics**: six-class confusion matrix, per-class
     precision/recall/F1, top-k accuracy, multiclass ECE.
  3. **Retrieval metrics**: Recall@k for k ∈ {1, 3, 5}, mean
     reciprocal rank (MRR), per-type recall. Relevance criterion:
     a retrieved case is relevant iff its crisis_type matches the
     query's. Hold-one-out evaluation: the query case is excluded
     from the retrieval pool.
  4. **MARL mitigation metrics**: mean episode reward, output-loss
     reduction vs. no-policy counterfactual baseline, safety-
     violation rate, case-coherence score (cosine similarity
     between executed actions and retrieved policy fingerprints).
  5. **Case-anchored synthetic control** (the central methodological
     contribution per the manuscript): for each test case, retrieve
     k anchor cases, construct a synthetic control trajectory as a
     weighted average of their post-onset paths using the
     retriever's softmax weights, and compute the treatment-effect
     estimate as the difference between the policy's rollout and
     the synthetic control. Reports per-case treatment effects with
     bootstrap confidence intervals (Abadie 2021).

Architecture
------------
::

    Evaluator(artefacts, library)
        │
        ├── evaluate_detection() ─────► detection_metrics: dict
        ├── evaluate_coordinator() ───► coordinator_metrics: dict
        ├── evaluate_retrieval() ─────► retrieval_metrics: dict
        ├── evaluate_mitigation() ────► mitigation_metrics: dict
        ├── evaluate_synthetic_control() ─► sc_metrics: dict
        │
        └── evaluate_all() ───────────► evaluation_manifest.json

Reproducibility commitments
---------------------------
* All RNG (used for negative-sample construction in detection
  evaluation and for bootstrap CIs) is seeded deterministically
  per-evaluator-instance from a single ``seed`` parameter.
* Per-evaluation-call seed offsets keep families' randomness
  independent: re-running coordinator eval with a different held-out
  split does not perturb retrieval eval.
* The output manifest is JSON with sorted keys, atomic writes, and
  ``allow_nan=False`` with explicit ``None`` sanitisation.
* CSV tables are written with deterministic column order and
  one-decimal-place rounding for direct LaTeX ``\\input{}``.

References (APA-7)
------------------
Abadie, A. (2021). Using synthetic controls: Feasibility, data
    requirements, and methodological aspects. Journal of Economic
    Literature, 59(2), 391-425.

Brier, G. W. (1950). Verification of forecasts expressed in terms
    of probability. Monthly Weather Review, 78(1), 1-3.

Davis, J., & Goadrich, M. (2006). The relationship between
    Precision-Recall and ROC curves. In Proceedings of the 23rd
    International Conference on Machine Learning (pp. 233-240).

Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining
    well calibrated probabilities using Bayesian binning. In
    Proceedings of the AAAI Conference on Artificial Intelligence
    (Vol. 29, pp. 2901-2907).

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is
    more informative than the ROC plot when evaluating binary
    classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432.

Version
-------
1.0.0  Camera-ready KBS submission.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Union

import numpy as np
import torch

from case_memory import (
    CRISIS_TYPES,
    CaseLibrary,
    CaseMemoryError,
    DEFAULT_FEATURE_NAMES,
    N_MACRO_FEATURES,
)
from analogy_engine import (
    DEFAULT_K_RETRIEVAL,
    DEFAULT_RETRIEVAL_TEMPERATURE,
    compute_policy_fingerprint,
)
from coordinator import COORDINATOR_TYPES
from detection import (
    DETECTOR_TYPES,
    POSITIVE_TYPES_PER_DETECTOR,
)
from mitigation import (
    AuthorityGraph,
    JOINT_ACTION_DIM,
    SafetyBounds,
    collapse_24dim_fp_to_per_lever,
    compute_reward,
)
from training import (
    TrainingArtifacts,
    TrainingError,
)


__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "DEFAULT_N_BOOTSTRAP",
    "DEFAULT_ECE_N_BINS",
    "DEFAULT_TOP_K",
    "OUTPUT_GAP_INDEX",
    # Exceptions
    "EvaluationError",
    "EmptyHoldoutError",
    # Helpers (pure functions)
    "auc_pr",
    "expected_calibration_error",
    "brier_score",
    "confusion_matrix",
    "precision_recall_f1_per_class",
    "top_k_accuracy",
    "multiclass_ece",
    "bootstrap_ci",
    # Evaluator
    "Evaluator",
    # Convenience
    "evaluate_from_directory",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Semantic version of the on-disk evaluation manifest.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: Default number of bootstrap resamples for confidence intervals.
DEFAULT_N_BOOTSTRAP: Final[int] = 1_000

#: Default number of bins for ECE. Naeini et al. (2015) recommend
#: 10-15 for typical probability ranges.
DEFAULT_ECE_N_BINS: Final[int] = 10

#: Default k values for Recall@k.
DEFAULT_TOP_K: Final[tuple[int, ...]] = (1, 3, 5)

#: Index of ``output_gap`` in ``DEFAULT_FEATURE_NAMES``. Computed at
#: import time to fail loudly if the schema drifts.
OUTPUT_GAP_INDEX: Final[int] = DEFAULT_FEATURE_NAMES.index("output_gap")

#: Index of ``inflation_yoy`` in ``DEFAULT_FEATURE_NAMES``.
INFLATION_YOY_INDEX: Final[int] = DEFAULT_FEATURE_NAMES.index("inflation_yoy")

#: Tolerance for "approximately zero" probability mass when checking
#: that posterior vectors sum to one.
_PROB_ATOL: Final[float] = 1e-5

#: Floor for log() arguments to avoid log(0) in NLL/ECE calculations.
_LOG_EPS: Final[float] = 1e-12


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class EvaluationError(CaseMemoryError):
    """Base class for evaluation-module exceptions."""


class EmptyHoldoutError(EvaluationError):
    """Held-out set is empty or degenerate (e.g., all one class)."""


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
    """Replace non-finite floats with None so the manifest can be
    serialised with ``allow_nan=False``.

    Recursive: handles dicts, lists/tuples, numpy arrays (converted
    to lists), numpy scalars (converted to Python scalars). Non-
    finite floats (NaN, ±inf) become ``None``.
    """
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (np.integer, int, bool)):
        return int(obj) if not isinstance(obj, bool) else bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitise_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise_for_json(v) for v in obj]
    return obj


def _validate_probabilities(
    p: np.ndarray, *, axis: int = -1, context: str = "probabilities",
) -> None:
    """Check that ``p`` is in [0, 1] elementwise and sums to ~1 along ``axis``.

    Used by ECE and confusion-matrix routines to fail loudly when a
    caller has passed logits or unnormalised scores.
    """
    if not np.all((p >= -_PROB_ATOL) & (p <= 1.0 + _PROB_ATOL)):
        raise EvaluationError(
            f"{context}: values must be in [0, 1]; got "
            f"min={float(p.min()):.4f}, max={float(p.max()):.4f}"
        )
    sums = p.sum(axis=axis)
    if not np.allclose(sums, 1.0, atol=_PROB_ATOL * max(1, p.shape[axis])):
        raise EvaluationError(
            f"{context}: sums along axis {axis} must be ~1.0; got "
            f"min={float(sums.min()):.4f}, max={float(sums.max()):.4f}"
        )


# ---------------------------------------------------------------------------
# Public metric helpers (pure functions)
# ---------------------------------------------------------------------------


def auc_pr(
    y_true: np.ndarray, y_score: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Area under the precision-recall curve via interpolation.

    Uses the average-precision estimator (Davis & Goadrich 2006):

    .. math::
        AP = \\sum_n (R_n - R_{n-1}) P_n

    where the precision/recall pairs are computed at each unique
    decreasing score threshold.

    Parameters
    ----------
    y_true : np.ndarray of shape (N,)
        Binary labels {0, 1}.
    y_score : np.ndarray of shape (N,)
        Predicted scores (higher = more positive).

    Returns
    -------
    ap : float
        Average precision in [0, 1].
    precision : np.ndarray of shape (N+1,)
    recall : np.ndarray of shape (N+1,)
        Aligned arrays of operating-point precision and recall.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise EvaluationError(
            f"auc_pr: shapes differ: {y_true.shape} vs {y_score.shape}"
        )
    if y_true.ndim != 1:
        raise EvaluationError(
            f"auc_pr: arrays must be 1-D; got {y_true.ndim}-D"
        )
    if not np.all(np.isin(y_true, [0, 1])):
        raise EvaluationError(
            "auc_pr: y_true must contain only 0 and 1"
        )
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        # No positives -> AP undefined; return 0 and flat curves.
        return 0.0, np.array([1.0, 0.0]), np.array([0.0, 0.0])
    # Sort by score descending
    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # Prepend (1, 0) for the empty-prediction operating point
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    # AP = sum of (R_n - R_{n-1}) * P_n
    ap = float(np.sum(np.diff(recall) * precision[1:]))
    return ap, precision, recall


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *,
    n_bins: int = DEFAULT_ECE_N_BINS,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Expected calibration error with equal-width binning.

    For each bin :math:`B_m` (defined on the predicted-probability
    axis), ECE is

    .. math::
        ECE = \\sum_m \\frac{|B_m|}{N} | \\text{acc}(B_m) - \\text{conf}(B_m) |

    where :math:`\\text{acc}` is the empirical accuracy of the bin
    and :math:`\\text{conf}` is the mean predicted probability.

    Parameters
    ----------
    y_true : np.ndarray of shape (N,)
        Binary labels.
    y_prob : np.ndarray of shape (N,)
        Predicted probabilities of class 1, in [0, 1].
    n_bins : int

    Returns
    -------
    ece : float
    bin_acc : np.ndarray of shape (n_bins,)
    bin_conf : np.ndarray of shape (n_bins,)
    bin_count : np.ndarray of shape (n_bins,)
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise EvaluationError(
            f"ECE: shapes differ: {y_true.shape} vs {y_prob.shape}"
        )
    if n_bins <= 0:
        raise EvaluationError(f"ECE: n_bins must be > 0; got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins, except the first which includes 0.
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False),
                       0, n_bins - 1)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_conf = np.zeros(n_bins, dtype=np.float64)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    for m in range(n_bins):
        mask = (bin_idx == m)
        c = int(mask.sum())
        bin_count[m] = c
        if c == 0:
            continue
        bin_acc[m] = float(y_true[mask].mean())
        bin_conf[m] = float(y_prob[mask].mean())
    total = float(bin_count.sum())
    if total == 0:
        return 0.0, bin_acc, bin_conf, bin_count
    ece = float(
        np.sum(
            bin_count.astype(np.float64) * np.abs(bin_acc - bin_conf)
        ) / total
    )
    return ece, bin_acc, bin_conf, bin_count


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean squared error between predicted probability
    and the binary label.

    Returns
    -------
    brier : float
        In [0, 1]. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise EvaluationError(
            f"brier_score: shapes differ: {y_true.shape} vs {y_prob.shape}"
        )
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, *, n_classes: int,
) -> np.ndarray:
    """Compute the confusion matrix.

    ``cm[i, j]`` counts samples with true label ``i`` and predicted
    label ``j``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise EvaluationError(
            f"confusion_matrix: shapes differ: {y_true.shape} vs "
            f"{y_pred.shape}"
        )
    if n_classes <= 0:
        raise EvaluationError(
            f"confusion_matrix: n_classes must be > 0; got {n_classes}"
        )
    if y_true.size > 0:
        if y_true.min() < 0 or y_true.max() >= n_classes:
            raise EvaluationError(
                f"confusion_matrix: y_true values out of range "
                f"[0, {n_classes}); got min={int(y_true.min())}, "
                f"max={int(y_true.max())}"
            )
        if y_pred.min() < 0 or y_pred.max() >= n_classes:
            raise EvaluationError(
                f"confusion_matrix: y_pred values out of range "
                f"[0, {n_classes}); got min={int(y_pred.min())}, "
                f"max={int(y_pred.max())}"
            )
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_per_class(
    cm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class precision, recall, and F1 from a confusion matrix.

    Classes with no support (zero true count) report recall=0, and
    classes with no predictions report precision=0. F1 follows the
    standard 2PR/(P+R) formula with the convention 0/0 = 0.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise EvaluationError(
            f"prec_recall_f1: cm must be square; got {cm.shape}"
        )
    diag = np.diag(cm)
    col_sum = cm.sum(axis=0)
    row_sum = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(col_sum > 0, diag / col_sum, 0.0)
        recall = np.where(row_sum > 0, diag / row_sum, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)
    return precision, recall, f1


def top_k_accuracy(
    y_true: np.ndarray, y_prob: np.ndarray, k: int,
) -> float:
    """Top-k classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray of shape (N,)
    y_prob : np.ndarray of shape (N, C)
        Probability matrix where row sums to 1.
    k : int
        Number of top predictions to consider.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.ndim != 2:
        raise EvaluationError(
            f"top_k_accuracy: bad shapes y_true={y_true.shape}, "
            f"y_prob={y_prob.shape}"
        )
    if y_true.shape[0] != y_prob.shape[0]:
        raise EvaluationError(
            f"top_k_accuracy: row-count mismatch {y_true.shape[0]} "
            f"vs {y_prob.shape[0]}"
        )
    if k <= 0:
        raise EvaluationError(f"top_k_accuracy: k must be > 0; got {k}")
    if k > y_prob.shape[1]:
        return 1.0  # k >= C means every label is in the top-k
    # Get indices of top-k classes per row
    top_k_idx = np.argpartition(-y_prob, k - 1, axis=1)[:, :k]
    correct = (top_k_idx == y_true[:, None]).any(axis=1)
    return float(correct.mean())


def multiclass_ece(
    y_true: np.ndarray, y_prob: np.ndarray, *,
    n_bins: int = DEFAULT_ECE_N_BINS,
) -> float:
    """Multiclass ECE based on the confidence (max-prob) decomposition.

    For each sample, compute confidence = max(prob) and correctness =
    (argmax(prob) == y_true). Then apply binary ECE on (correctness,
    confidence) pairs.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim != 2:
        raise EvaluationError(
            f"multiclass_ece: y_prob must be 2-D; got {y_prob.shape}"
        )
    _validate_probabilities(y_prob, axis=-1, context="multiclass_ece y_prob")
    confidences = y_prob.max(axis=-1)
    predictions = y_prob.argmax(axis=-1)
    correct = (predictions == y_true).astype(np.int64)
    ece, _, _, _ = expected_calibration_error(
        correct, confidences, n_bins=n_bins,
    )
    return ece


def bootstrap_ci(
    values: np.ndarray, *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = 0,
    statistic: str = "mean",
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a univariate statistic.

    Parameters
    ----------
    values : np.ndarray of shape (N,)
    n_bootstrap : int
        Number of resamples.
    confidence : float
        E.g. 0.95 for a 95% CI.
    seed : int
    statistic : str
        One of {"mean", "median"}.

    Returns
    -------
    point : float
        Statistic computed on ``values``.
    lo : float
    hi : float
        Lower and upper percentile bounds of the bootstrap
        distribution.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise EvaluationError(
            f"bootstrap_ci: values must be 1-D; got {values.shape}"
        )
    n = values.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if statistic == "mean":
        stat_fn = np.mean
    elif statistic == "median":
        stat_fn = np.median
    else:
        raise EvaluationError(
            f"bootstrap_ci: unknown statistic {statistic!r}"
        )
    point = float(stat_fn(values))
    if n == 1:
        # Degenerate: the bootstrap distribution is a point mass.
        return point, point, point
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n, size=(n_bootstrap, n))
    bs_stats = np.array([float(stat_fn(values[s])) for s in samples])
    alpha = 1.0 - float(confidence)
    lo = float(np.percentile(bs_stats, 100 * alpha / 2))
    hi = float(np.percentile(bs_stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@dataclass
class _DetectionEvalResult:
    per_detector: dict[str, dict[str, float]]
    macro_auc_pr: float
    macro_ece: float
    macro_brier: float


@dataclass
class _CoordinatorEvalResult:
    confusion_matrix: np.ndarray
    per_class: dict[str, dict[str, float]]
    top_1_accuracy: float
    top_2_accuracy: float
    multiclass_ece: float
    macro_f1: float
    n_samples: int


@dataclass
class _RetrievalEvalResult:
    recall_at_k: dict[int, float]
    mrr: float
    per_type_recall_at_1: dict[str, float]
    n_queries: int


@dataclass
class _MitigationEvalResult:
    mean_episode_reward: float
    output_loss_reduction_pct: float
    safety_violation_rate: float
    case_coherence_mean: float
    n_episodes: int


@dataclass
class _SyntheticControlEvalResult:
    mean_treatment_effect: float
    ci_low: float
    ci_high: float
    n_cases: int
    per_type_effects: dict[str, float]


class Evaluator:
    """Five-family metric evaluator.

    Parameters
    ----------
    artefacts : TrainingArtifacts
        Populated by ``training.TrainingOrchestrator.run_all_phases``.
        Must have ``is_complete=True``.
    library : CaseLibrary
        Cases to evaluate on. May be the same library used for
        training (in-sample evaluation), a held-out library, or a
        mixture; the manifest records which library was used via its
        checksum.
    seed : int
        Master seed; per-family seeds derived deterministically.
    device : str or torch.device
        Where to run forward passes.
    """

    PHASE_SEEDS: Final[dict[str, int]] = {
        "detection":     100,
        "coordinator":   200,
        "retrieval":     300,
        "mitigation":    400,
        "synthetic_control": 500,
    }

    def __init__(
        self,
        artefacts: TrainingArtifacts,
        library: CaseLibrary,
        *,
        seed: int = 42,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if not isinstance(artefacts, TrainingArtifacts):
            raise EvaluationError(
                f"Evaluator: artefacts must be TrainingArtifacts; "
                f"got {type(artefacts)}"
            )
        if not artefacts.is_complete:
            raise EvaluationError(
                "Evaluator: artefacts.is_complete must be True; "
                "some phase was not trained or loaded"
            )
        if not isinstance(library, CaseLibrary):
            raise EvaluationError(
                f"Evaluator: library must be CaseLibrary; "
                f"got {type(library)}"
            )
        if len(library) == 0:
            raise EvaluationError("Evaluator: library is empty")
        self.artefacts: TrainingArtifacts = artefacts
        self.library: CaseLibrary = library
        self.seed: int = int(seed)
        self.device: torch.device = torch.device(device)
        # Eval mode for all torch modules
        for m in (
            artefacts.detection_ensemble,
            artefacts.coordinator,
            artefacts.signature_encoder,
            artefacts.mitigation_policy,
            artefacts.dynamics,
        ):
            if isinstance(m, torch.nn.Module):
                m.eval()
        # Library checksum for the manifest (so a stale eval against
        # a new library is detectable).
        case_ids_sorted = sorted(library.case_ids())
        self._library_checksum: str = _sha256_hex(
            "\n".join(case_ids_sorted)
        )[:16]
        # Internal results cache (populated by evaluate_* methods)
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _family_seed(self, family: str) -> int:
        if family not in self.PHASE_SEEDS:
            raise EvaluationError(
                f"_family_seed: unknown family {family!r}"
            )
        return self.seed + self.PHASE_SEEDS[family]

    # ------------------------------------------------------------------ #
    # Family 1: Detection
    # ------------------------------------------------------------------ #

    def evaluate_detection(self) -> dict[str, Any]:
        """Evaluate the four binary crisis detectors.

        For each detector, we use the pre-onset trajectories of all
        library cases as positives (label = 1 iff the case's crisis
        type is in ``POSITIVE_TYPES_PER_DETECTOR[detector]``) and
        generate the same count of synthetic negatives by sampling
        random non-crisis trajectories from the early-history
        window of each case.

        Reports per-detector AUC-PR, ECE@10-bins, Brier score; and
        the macro-average across detectors.
        """
        seed = self._family_seed("detection")
        ensemble = self.artefacts.detection_ensemble
        if ensemble is None:
            raise EvaluationError("evaluate_detection: no ensemble")
        rng = np.random.default_rng(seed)

        case_ids = self.library.case_ids()
        n_cases = len(case_ids)
        # Build positive trajectories and their crisis types
        pos_trajectories: list[np.ndarray] = []
        pos_types: list[str] = []
        for cid in case_ids:
            case = self.library[cid]
            pos_trajectories.append(case.pre_onset_trajectory.copy())
            pos_types.append(case.crisis_type)
        # Build negative trajectories: shuffle the order of pre-onset
        # quarters within each case to destroy the crisis signature,
        # producing a same-distribution non-crisis trajectory. This is
        # the same negative-construction strategy used in
        # detection.DetectorEnsemble.fit's synthetic-negatives path.
        neg_trajectories: list[np.ndarray] = []
        for cid in case_ids:
            case = self.library[cid]
            T = case.pre_onset_trajectory.shape[0]
            perm = rng.permutation(T)
            neg_trajectories.append(case.pre_onset_trajectory[perm].copy())

        all_trajectories = pos_trajectories + neg_trajectories
        is_pos = np.array(
            [1] * n_cases + [0] * n_cases, dtype=np.int64,
        )
        all_types = pos_types + ["__negative__"] * n_cases

        # Per-trajectory probabilities from the ensemble
        # detect() returns a DetectionOutput with shape (n_detectors,)
        probs_matrix = np.zeros(
            (len(all_trajectories), len(DETECTOR_TYPES)), dtype=np.float64,
        )
        for i, traj in enumerate(all_trajectories):
            out = ensemble.detect(traj)
            probs_matrix[i] = out.probabilities

        # Per-detector eval
        per_detector: dict[str, dict[str, float]] = {}
        macro_ap, macro_ece, macro_brier = [], [], []
        for j, det_name in enumerate(DETECTOR_TYPES):
            # Per-detector label: positive iff (the trajectory is a
            # positive sample AND its crisis type is in the
            # detector's positive set)
            pos_types_for_det = POSITIVE_TYPES_PER_DETECTOR[det_name]
            y = np.zeros(len(all_trajectories), dtype=np.int64)
            for i, (is_p, t) in enumerate(zip(is_pos, all_types)):
                if is_p and t in pos_types_for_det:
                    y[i] = 1
            scores = probs_matrix[:, j]
            ap, _, _ = auc_pr(y, scores)
            ece_val, _, _, _ = expected_calibration_error(y, scores)
            br = brier_score(y, scores)
            per_detector[det_name] = {
                "auc_pr": float(ap),
                "ece": float(ece_val),
                "brier": float(br),
                "n_positive": int(y.sum()),
                "n_total": int(y.size),
            }
            macro_ap.append(ap)
            macro_ece.append(ece_val)
            macro_brier.append(br)

        result_dict = {
            "per_detector": per_detector,
            "macro_auc_pr": float(np.mean(macro_ap)),
            "macro_ece": float(np.mean(macro_ece)),
            "macro_brier": float(np.mean(macro_brier)),
            "n_cases_evaluated": n_cases,
            "seed": seed,
        }
        self._cache["detection"] = result_dict
        return result_dict

    # ------------------------------------------------------------------ #
    # Family 2: Coordinator
    # ------------------------------------------------------------------ #

    def evaluate_coordinator(self) -> dict[str, Any]:
        """Evaluate the coordinator.

        For each library case, compute the detector probabilities and
        macro state (last pre-onset quarter), call the coordinator,
        and compare its argmax to the case's true crisis type. Reports
        confusion matrix, per-class P/R/F1, top-1 and top-2 accuracy,
        multiclass ECE.

        Note that ``COORDINATOR_TYPES`` includes ``'none'`` as class 0,
        but library cases never have ``crisis_type='none'``, so the
        confusion matrix's row 0 will always be empty.
        """
        seed = self._family_seed("coordinator")
        coordinator = self.artefacts.coordinator
        ensemble = self.artefacts.detection_ensemble
        if coordinator is None or ensemble is None:
            raise EvaluationError(
                "evaluate_coordinator: missing coordinator or ensemble"
            )

        case_ids = self.library.case_ids()
        n = len(case_ids)
        n_classes = len(COORDINATOR_TYPES)
        y_true = np.zeros(n, dtype=np.int64)
        y_prob = np.zeros((n, n_classes), dtype=np.float64)

        for i, cid in enumerate(case_ids):
            case = self.library[cid]
            det_out = ensemble.detect(case.pre_onset_trajectory)
            macro_state = case.pre_onset_trajectory[-1]
            coord_out = coordinator.coordinate(
                macro_state, det_out.probabilities,
            )
            y_true[i] = COORDINATOR_TYPES.index(case.crisis_type)
            y_prob[i] = coord_out.posterior

        y_pred = y_prob.argmax(axis=1)
        cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
        precision, recall, f1 = precision_recall_f1_per_class(cm)

        per_class: dict[str, dict[str, float]] = {}
        for k, name in enumerate(COORDINATOR_TYPES):
            per_class[name] = {
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "support": int(cm[k].sum()),
            }

        top1 = top_k_accuracy(y_true, y_prob, k=1)
        top2 = top_k_accuracy(y_true, y_prob, k=2)
        mc_ece = multiclass_ece(y_true, y_prob)

        # Macro F1 ignoring zero-support classes
        valid = cm.sum(axis=1) > 0
        macro_f1 = float(np.mean(f1[valid])) if valid.any() else 0.0

        result_dict = {
            "confusion_matrix": cm.tolist(),
            "class_names": list(COORDINATOR_TYPES),
            "per_class": per_class,
            "top_1_accuracy": float(top1),
            "top_2_accuracy": float(top2),
            "multiclass_ece": float(mc_ece),
            "macro_f1": float(macro_f1),
            "n_samples": int(n),
            "seed": seed,
        }
        self._cache["coordinator"] = result_dict
        return result_dict

    # ------------------------------------------------------------------ #
    # Family 3: Retrieval
    # ------------------------------------------------------------------ #

    def evaluate_retrieval(
        self, *, k_values: Sequence[int] = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        """Evaluate the analogy engine via leave-one-out retrieval.

        For each case in the library, treat it as a query, retrieve
        the top-k cases from the rest of the library, and consider a
        retrieved case "relevant" iff its crisis_type matches the
        query's.

        Reports Recall@k for each k, MRR (1 / rank of the first
        relevant case, averaged across queries; 0 if no relevant case
        appears in top-K), and per-type Recall@1.

        Implementation note: since ``AnalogyEngine`` is built against
        the full library at construction time, we cannot trivially
        exclude the query case from its index. We work around this by
        running the engine's normal retrieve and filtering out the
        query's own ID from the results, then truncating to k. To
        get k results after filtering, we request k+1.
        """
        seed = self._family_seed("retrieval")
        engine = self.artefacts.analogy_engine
        if engine is None:
            raise EvaluationError("evaluate_retrieval: no analogy engine")

        case_ids = list(self.library.case_ids())
        if len(case_ids) < 2:
            raise EmptyHoldoutError(
                f"evaluate_retrieval: need >= 2 cases; got {len(case_ids)}"
            )
        max_k = max(k_values)
        k_values_sorted = sorted(set(int(k) for k in k_values))

        # For each query, retrieve max_k + 1 cases (to allow excluding
        # the self-match), filter, and record the rank of each
        # same-type relevant retrieval.
        recall_at_k: dict[int, list[float]] = {k: [] for k in k_values_sorted}
        first_relevant_rank: list[float] = []  # 1-indexed; inf if no relevant
        per_type_recall_at_1: dict[str, list[float]] = {
            ct: [] for ct in CRISIS_TYPES
        }

        for cid in case_ids:
            case = self.library[cid]
            # Ground-truth type posterior, since we're in oracle mode
            # (we want to evaluate the retriever, not the coordinator).
            tp = np.zeros(len(CRISIS_TYPES), dtype=np.float64)
            tp[CRISIS_TYPES.index(case.crisis_type)] = 1.0
            result = engine.retrieve(
                case.pre_onset_trajectory.copy(),
                tp,
                k=max_k + 1,
                temperature=DEFAULT_RETRIEVAL_TEMPERATURE,
            )
            # Filter out self
            retrieved = [r for r in result.case_ids if r != cid]
            retrieved = retrieved[:max_k]
            # Compute per-k recall: fraction of relevant cases in top-k
            # over the number of relevant cases in the pool.
            n_relevant_in_pool = sum(
                1 for other_cid in case_ids
                if other_cid != cid
                and self.library[other_cid].crisis_type == case.crisis_type
            )
            if n_relevant_in_pool == 0:
                # Singleton-type case: skip
                continue
            # Recall@k: fraction of relevant items retrieved in top-k.
            # Capped at 1.0 (in case k > n_relevant_in_pool).
            for k in k_values_sorted:
                top_k = retrieved[:k]
                n_relevant_in_top_k = sum(
                    1 for r in top_k
                    if self.library[r].crisis_type == case.crisis_type
                )
                recall_at_k[k].append(
                    n_relevant_in_top_k / n_relevant_in_pool
                )
            # MRR: record the 1-indexed rank of the first relevant
            # retrieval (inf if none). The actual MRR statistic is
            # computed across all queries in the aggregation step.
            for rank, r in enumerate(retrieved, start=1):
                if self.library[r].crisis_type == case.crisis_type:
                    first_relevant_rank.append(float(rank))
                    break
            else:
                first_relevant_rank.append(float("inf"))
            # Per-type recall@1
            top1 = retrieved[0] if retrieved else None
            top1_relevant = (
                top1 is not None
                and self.library[top1].crisis_type == case.crisis_type
            )
            per_type_recall_at_1[case.crisis_type].append(
                1.0 if top1_relevant else 0.0
            )

        # Aggregate
        recall_at_k_mean: dict[int, float] = {}
        for k, vals in recall_at_k.items():
            recall_at_k_mean[k] = float(np.mean(vals)) if vals else float("nan")
        # MRR
        finite_mrr = [
            1.0 / r for r in first_relevant_rank if math.isfinite(r)
        ]
        mrr = (
            float(np.mean(finite_mrr + [0.0] * (len(first_relevant_rank) - len(finite_mrr))))
            if first_relevant_rank else float("nan")
        )
        per_type_recall = {
            ct: (float(np.mean(vals)) if vals else float("nan"))
            for ct, vals in per_type_recall_at_1.items()
        }

        result_dict = {
            "recall_at_k": {str(k): v for k, v in recall_at_k_mean.items()},
            "mrr": mrr,
            "per_type_recall_at_1": per_type_recall,
            "n_queries": len(case_ids),
            "k_values": list(k_values_sorted),
            "seed": seed,
        }
        self._cache["retrieval"] = result_dict
        return result_dict

    # ------------------------------------------------------------------ #
    # Family 4: Mitigation
    # ------------------------------------------------------------------ #

    def evaluate_mitigation(
        self, *,
        n_episodes: int = 50,
        episode_len: int = 12,
        k_retrieval: int = DEFAULT_K_RETRIEVAL,
        apply_cbf: bool = True,
    ) -> dict[str, Any]:
        """Evaluate the mitigation policy via rollouts.

        For each of ``n_episodes`` evaluation episodes, sample a
        starting case (with deterministic seeding), run the policy
        for ``episode_len`` steps with exploration noise = 0 (greedy),
        and record:

          * Episode reward (sum of per-step rewards).
          * Whether any state violated ``SafetyBounds``.
          * Cosine similarity between the executed action and the
            retrieved-cases' policy fingerprint mean
            (case-coherence score).

        Parameters
        ----------
        apply_cbf : bool
            Whether to apply the control-barrier-function safety
            projection during rollouts. Set False for ablation
            studies that measure the contribution of CBF (otherwise
            keep the camera-ready default of True).

        Then run a no-policy counterfactual: the same starting state,
        but with zero actions throughout. Compare the cumulative
        output-gap loss (sum of |output_gap|) under policy vs.
        counterfactual, and report the relative reduction.
        """
        seed = self._family_seed("mitigation")
        policy = self.artefacts.mitigation_policy
        dynamics = self.artefacts.dynamics
        engine = self.artefacts.analogy_engine
        if policy is None or dynamics is None or engine is None:
            raise EvaluationError(
                "evaluate_mitigation: missing policy, dynamics, or engine"
            )
        authority = AuthorityGraph.from_dict(policy.authority.to_dict())
        safety_bounds = SafetyBounds()  # default bounds
        rng = np.random.default_rng(seed)
        case_ids = list(self.library.case_ids())

        episode_rewards: list[float] = []
        violation_counts: list[float] = []
        coherence_scores: list[float] = []
        policy_output_losses: list[float] = []
        nopolicy_output_losses: list[float] = []

        for ep in range(n_episodes):
            cid = case_ids[rng.integers(0, len(case_ids))]
            case = self.library[cid]
            initial_state = case.pre_onset_trajectory[-1].astype(np.float64)
            query_traj = case.pre_onset_trajectory.copy()
            tp = np.zeros(len(CRISIS_TYPES), dtype=np.float64)
            tp[CRISIS_TYPES.index(case.crisis_type)] = 1.0

            # Policy rollout
            state = initial_state.copy()
            rewards: list[float] = []
            violations_this_ep = 0
            coherences_this_ep: list[float] = []
            policy_cum_loss = 0.0
            for step in range(episode_len):
                ret = engine.retrieve(
                    query_traj.copy(), tp, k=k_retrieval,
                    temperature=DEFAULT_RETRIEVAL_TEMPERATURE,
                )
                action_unit, _ = policy.get_action(
                    state, ret, exploration_noise=0.0, apply_cbf=apply_cbf,
                )
                # Safety check (state before action)
                if not safety_bounds.all_satisfied(state):
                    violations_this_ep += 1
                # Case-coherence: cosine similarity between
                # executed action and the retrieved cases'
                # weighted-mean per-lever policy fingerprint.
                fps = np.stack([
                    collapse_24dim_fp_to_per_lever(
                        compute_policy_fingerprint(self.library[rcid])
                    )
                    for rcid in ret.case_ids
                ], axis=0)  # (K, 8)
                # Weight by retrieval weights; standardise fps to
                # the same scale as the unit action.
                fp_mean_phys = (
                    ret.weights[:, None] * fps
                ).sum(axis=0)  # (8,) in physical units
                fp_mean_unit = authority.scale_to_unit(fp_mean_phys)
                # Cosine similarity; guard against zero norm
                num = float(np.dot(action_unit, fp_mean_unit))
                den = float(
                    np.linalg.norm(action_unit)
                    * np.linalg.norm(fp_mean_unit)
                )
                cos_sim = num / den if den > _LOG_EPS else 0.0
                coherences_this_ep.append(cos_sim)

                # Step dynamics
                physical_action = authority.scale_to_physical(action_unit)
                state_t = torch.as_tensor(
                    state, dtype=torch.float32, device=self.device,
                )
                phys_t = torch.as_tensor(
                    physical_action, dtype=torch.float32, device=self.device,
                )
                with torch.no_grad():
                    next_state_t = dynamics.step(state_t, phys_t)
                next_state = next_state_t.cpu().numpy().astype(np.float64)
                # Reward
                r = compute_reward(state, action_unit)
                rewards.append(float(r))
                # Output-gap loss
                policy_cum_loss += abs(float(state[OUTPUT_GAP_INDEX]))
                # Advance
                query_traj = np.concatenate(
                    [query_traj[1:], next_state[None, :]], axis=0,
                )
                state = next_state
            # Final state's output-gap contribution
            policy_cum_loss += abs(float(state[OUTPUT_GAP_INDEX]))

            # No-policy counterfactual
            state_np = initial_state.copy()
            nopol_cum_loss = 0.0
            zero_action_unit = np.zeros(JOINT_ACTION_DIM, dtype=np.float64)
            zero_action_phys = authority.scale_to_physical(zero_action_unit)
            for step in range(episode_len):
                state_t = torch.as_tensor(
                    state_np, dtype=torch.float32, device=self.device,
                )
                phys_t = torch.as_tensor(
                    zero_action_phys, dtype=torch.float32, device=self.device,
                )
                with torch.no_grad():
                    next_state_t = dynamics.step(state_t, phys_t)
                nopol_cum_loss += abs(float(state_np[OUTPUT_GAP_INDEX]))
                state_np = next_state_t.cpu().numpy().astype(np.float64)
            nopol_cum_loss += abs(float(state_np[OUTPUT_GAP_INDEX]))

            episode_rewards.append(float(sum(rewards)))
            violation_counts.append(
                float(violations_this_ep) / float(episode_len)
            )
            coherence_scores.append(
                float(np.mean(coherences_this_ep)) if coherences_this_ep else 0.0
            )
            policy_output_losses.append(policy_cum_loss)
            nopolicy_output_losses.append(nopol_cum_loss)

        # Output-loss reduction (positive = policy reduces loss)
        pol_arr = np.array(policy_output_losses)
        np_arr = np.array(nopolicy_output_losses)
        # Avoid div-by-zero
        with np.errstate(divide="ignore", invalid="ignore"):
            per_ep_reduction = np.where(
                np_arr > _LOG_EPS,
                (np_arr - pol_arr) / np_arr * 100.0,
                0.0,
            )

        result_dict = {
            "mean_episode_reward": float(np.mean(episode_rewards)),
            "std_episode_reward": float(np.std(episode_rewards)),
            "mean_safety_violation_rate": float(np.mean(violation_counts)),
            "mean_case_coherence": float(np.mean(coherence_scores)),
            "mean_output_loss_policy": float(np.mean(pol_arr)),
            "mean_output_loss_nopolicy": float(np.mean(np_arr)),
            "mean_output_loss_reduction_pct": float(np.mean(per_ep_reduction)),
            "n_episodes": int(n_episodes),
            "episode_len": int(episode_len),
            "k_retrieval": int(k_retrieval),
            "seed": seed,
        }
        self._cache["mitigation"] = result_dict
        return result_dict

    # ------------------------------------------------------------------ #
    # Family 5: Case-Anchored Synthetic Control
    # ------------------------------------------------------------------ #

    def evaluate_synthetic_control(
        self, *,
        k_retrieval: int = DEFAULT_K_RETRIEVAL,
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
        apply_cbf: bool = True,
    ) -> dict[str, Any]:
        """Case-anchored synthetic-control treatment-effect estimation.

        For each case in the library:
          1. Retrieve k anchor cases (excluding self).
          2. Build a synthetic-control trajectory as the
             retrieval-weighted average of the anchors' post-onset
             trajectories.
          3. Roll out the policy on the dynamics starting from the
             case's pre-onset endpoint, for the same horizon as the
             post-onset window.
          4. Treatment effect = cumulative |output_gap| under the
             synthetic control MINUS cumulative |output_gap| under
             the policy rollout. Positive = policy outperforms
             counterfactual.

        Reports the per-case effect with bootstrap CI, and per-type
        means.
        """
        seed = self._family_seed("synthetic_control")
        policy = self.artefacts.mitigation_policy
        dynamics = self.artefacts.dynamics
        engine = self.artefacts.analogy_engine
        if policy is None or dynamics is None or engine is None:
            raise EvaluationError(
                "evaluate_synthetic_control: missing policy, dynamics, "
                "or engine"
            )
        authority = AuthorityGraph.from_dict(policy.authority.to_dict())
        case_ids = list(self.library.case_ids())
        if len(case_ids) < 2:
            raise EmptyHoldoutError(
                f"evaluate_synthetic_control: need >= 2 cases; "
                f"got {len(case_ids)}"
            )

        treatment_effects: list[float] = []
        per_type: dict[str, list[float]] = {ct: [] for ct in CRISIS_TYPES}

        for cid in case_ids:
            case = self.library[cid]
            initial_state = case.pre_onset_trajectory[-1].astype(np.float64)
            horizon = case.post_onset_trajectory.shape[0]
            query_traj = case.pre_onset_trajectory.copy()
            tp = np.zeros(len(CRISIS_TYPES), dtype=np.float64)
            tp[CRISIS_TYPES.index(case.crisis_type)] = 1.0

            # Retrieve k+1 then filter self
            ret = engine.retrieve(
                query_traj, tp, k=k_retrieval + 1,
                temperature=DEFAULT_RETRIEVAL_TEMPERATURE,
            )
            anchors = [
                (rcid, w)
                for rcid, w in zip(ret.case_ids, ret.weights)
                if rcid != cid
            ][:k_retrieval]
            if not anchors:
                continue
            # Renormalise weights (since self may have been dropped)
            w_total = sum(w for _, w in anchors)
            if w_total <= _LOG_EPS:
                continue

            # Build synthetic-control trajectory: weighted mean of
            # anchors' post-onset trajectories. Truncate or pad to
            # `horizon` quarters.
            sc_traj = np.zeros(
                (horizon, N_MACRO_FEATURES), dtype=np.float64,
            )
            for rcid, w in anchors:
                anchor_post = self.library[rcid].post_onset_trajectory
                T = min(horizon, anchor_post.shape[0])
                sc_traj[:T] += (w / w_total) * anchor_post[:T]
            sc_cum_loss = float(
                np.sum(np.abs(sc_traj[:, OUTPUT_GAP_INDEX]))
            )

            # Policy rollout for `horizon` quarters
            state = initial_state.copy()
            policy_cum_loss = 0.0
            qt = query_traj.copy()
            for step in range(horizon):
                ret_step = engine.retrieve(
                    qt.copy(), tp, k=k_retrieval,
                    temperature=DEFAULT_RETRIEVAL_TEMPERATURE,
                )
                action_unit, _ = policy.get_action(
                    state, ret_step, exploration_noise=0.0, apply_cbf=apply_cbf,
                )
                physical_action = authority.scale_to_physical(action_unit)
                state_t = torch.as_tensor(
                    state, dtype=torch.float32, device=self.device,
                )
                phys_t = torch.as_tensor(
                    physical_action, dtype=torch.float32, device=self.device,
                )
                with torch.no_grad():
                    next_state_t = dynamics.step(state_t, phys_t)
                next_state = next_state_t.cpu().numpy().astype(np.float64)
                policy_cum_loss += abs(float(state[OUTPUT_GAP_INDEX]))
                qt = np.concatenate(
                    [qt[1:], next_state[None, :]], axis=0,
                )
                state = next_state

            # Treatment effect: SC - policy. Positive = policy wins.
            te = sc_cum_loss - policy_cum_loss
            treatment_effects.append(te)
            per_type[case.crisis_type].append(te)

        if not treatment_effects:
            raise EmptyHoldoutError(
                "evaluate_synthetic_control: no usable cases (all had "
                "degenerate retrieval weights)"
            )

        effects_arr = np.array(treatment_effects, dtype=np.float64)
        point, lo, hi = bootstrap_ci(
            effects_arr, n_bootstrap=n_bootstrap, confidence=0.95,
            seed=seed, statistic="mean",
        )

        result_dict = {
            "mean_treatment_effect": point,
            "ci_low_95": lo,
            "ci_high_95": hi,
            "median_treatment_effect": float(np.median(effects_arr)),
            "per_type_mean_treatment_effect": {
                ct: (float(np.mean(v)) if v else float("nan"))
                for ct, v in per_type.items()
            },
            "n_cases": int(effects_arr.size),
            "n_bootstrap": int(n_bootstrap),
            "per_case_treatment_effects": effects_arr.tolist(),
            "seed": seed,
        }
        self._cache["synthetic_control"] = result_dict
        return result_dict

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def evaluate_all(
        self, *,
        mitigation_n_episodes: int = 50,
        mitigation_episode_len: int = 12,
        k_retrieval: int = DEFAULT_K_RETRIEVAL,
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
        apply_cbf: bool = True,
        save_to: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Run all five metric families and (optionally) save a manifest.

        Parameters
        ----------
        apply_cbf : bool
            Threaded through to the mitigation and synthetic-control
            evaluators. Set False for the no-CBF ablation.

        Returns
        -------
        manifest : dict[str, Any]
            JSON-serialisable evaluation manifest.
        """
        logger.info(
            "Evaluator: starting evaluate_all on %d cases (device=%s)",
            len(self.library), self.device,
        )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "library_checksum": self._library_checksum,
            "library_size": int(len(self.library)),
            "device": str(self.device),
            "seed": int(self.seed),
            "started_at": _utc_now_iso(),
        }
        with torch.no_grad():
            manifest["detection"] = self.evaluate_detection()
            logger.info("Evaluator: detection done")
            manifest["coordinator"] = self.evaluate_coordinator()
            logger.info("Evaluator: coordinator done")
            manifest["retrieval"] = self.evaluate_retrieval()
            logger.info("Evaluator: retrieval done")
            manifest["mitigation"] = self.evaluate_mitigation(
                n_episodes=mitigation_n_episodes,
                episode_len=mitigation_episode_len,
                k_retrieval=k_retrieval,
                apply_cbf=apply_cbf,
            )
            logger.info("Evaluator: mitigation done")
            manifest["synthetic_control"] = self.evaluate_synthetic_control(
                k_retrieval=k_retrieval,
                n_bootstrap=n_bootstrap,
                apply_cbf=apply_cbf,
            )
            logger.info("Evaluator: synthetic_control done")
        manifest["finished_at"] = _utc_now_iso()
        manifest = _sanitise_for_json(manifest)
        if save_to is not None:
            self.save_manifest(manifest, save_to)
        return manifest

    @staticmethod
    def save_manifest(manifest: dict[str, Any], path: Path) -> Path:
        """Atomically write the evaluation manifest to ``path``."""
        path = Path(path)
        _atomic_write_text(
            path,
            json.dumps(manifest, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        return path

    def summary(self) -> dict[str, Any]:
        """Brief summary of completed evaluations."""
        return {
            "schema_version": SCHEMA_VERSION,
            "library_checksum": self._library_checksum,
            "library_size": int(len(self.library)),
            "completed_families": sorted(self._cache.keys()),
            "device": str(self.device),
            "seed": int(self.seed),
        }

    def __repr__(self) -> str:
        return (
            f"Evaluator(library={self._library_checksum}, "
            f"completed={sorted(self._cache.keys())}, "
            f"device={self.device})"
        )


# ---------------------------------------------------------------------------
# Convenience: load artefacts and evaluate from a directory
# ---------------------------------------------------------------------------


def evaluate_from_directory(
    training_dir: Path,
    library: CaseLibrary,
    *,
    seed: int = 42,
    device: Union[str, torch.device] = "cpu",
    save_manifest: bool = True,
    **eval_kwargs: Any,
) -> dict[str, Any]:
    """Load a completed training run from ``training_dir`` and evaluate.

    Convenience wrapper for the common workflow:
    ``TrainingOrchestrator`` → save → re-load in a separate process →
    evaluate. Requires that all five phases completed successfully.

    Saves the evaluation manifest to
    ``training_dir / 'evaluation_manifest.json'`` if
    ``save_manifest=True``.
    """
    from training import TrainingOrchestrator, TrainingConfig
    training_dir = Path(training_dir)
    if not (training_dir / "training_manifest.json").is_file():
        raise EvaluationError(
            f"evaluate_from_directory: no training_manifest.json at "
            f"{training_dir}"
        )
    manifest = json.loads(
        (training_dir / "training_manifest.json").read_text(encoding="utf-8")
    )
    if not manifest.get("is_complete"):
        raise EvaluationError(
            f"evaluate_from_directory: training at {training_dir} is "
            f"not complete (is_complete=False)"
        )
    # Rebuild a TrainingConfig from the manifest's config dict
    cfg_dict = dict(manifest["config"])
    cfg_dict["output_dir"] = Path(cfg_dict["output_dir"])
    cfg = TrainingConfig(**cfg_dict)
    # Reconstruct orchestrator and reload artefacts
    orchestrator = TrainingOrchestrator(cfg, library)
    orchestrator.run_all_phases(force_retrain=False)
    if not orchestrator.artefacts.is_complete:
        raise EvaluationError(
            "evaluate_from_directory: failed to reload all artefacts"
        )
    evaluator = Evaluator(
        orchestrator.artefacts, library, seed=seed, device=device,
    )
    save_to = (
        training_dir / "evaluation_manifest.json" if save_manifest else None
    )
    return evaluator.evaluate_all(save_to=save_to, **eval_kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> int:
    """CLI: ``python evaluation.py <command> [args]``.

    Commands
    --------
    synthetic <training_dir> [--n-per-type N] [--seed S]
        Run evaluation on a synthetic library matching the training
        config. Convenient smoke test.
    summary <manifest_path>
        Print an evaluation manifest.
    """
    import argparse
    from data_pipeline import build_synthetic_library

    parser = argparse.ArgumentParser(
        prog="python evaluation.py",
        description="Evaluate a trained framework run.",
    )
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_syn = sub.add_parser(
        "synthetic", help="Evaluate a synthetic-library training run.",
    )
    p_syn.add_argument("training_dir", type=Path)
    p_syn.add_argument("--n-per-type", type=int, default=5)
    p_syn.add_argument("--seed", type=int, default=42)
    p_syn.add_argument("--device", default="cpu")
    p_syn.add_argument("--quick", action="store_true",
                       help="Use small episode counts for a smoke test.")

    p_sum = sub.add_parser("summary", help="Print an evaluation manifest.")
    p_sum.add_argument("manifest_path", type=Path)

    args = parser.parse_args()
    _configure_logging(args.log_level)

    try:
        if args.command == "synthetic":
            library = build_synthetic_library(
                n_per_type=args.n_per_type, seed=args.seed,
            )
            kwargs: dict[str, Any] = {}
            if args.quick:
                kwargs.update(dict(
                    mitigation_n_episodes=4,
                    mitigation_episode_len=4,
                    n_bootstrap=100,
                ))
            manifest = evaluate_from_directory(
                args.training_dir, library,
                seed=args.seed, device=args.device,
                **kwargs,
            )
            print(json.dumps(
                {k: v for k, v in manifest.items()
                 if k != "synthetic_control"
                 or not isinstance(v, dict)
                 or "per_case_treatment_effects" not in v},
                indent=2, sort_keys=True,
            ))
            return 0
        elif args.command == "summary":
            manifest = json.loads(
                args.manifest_path.read_text(encoding="utf-8")
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        else:  # pragma: no cover
            parser.print_help()
            return 2
    except (CaseMemoryError, TrainingError, EvaluationError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
