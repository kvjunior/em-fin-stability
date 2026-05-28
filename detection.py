"""
detection.py
============

Per-type crisis detectors: the front-end of the inference pipeline.

This module sits at the top of the inference graph: it consumes a
20-quarter pre-onset macro trajectory and emits four marginal
crisis-onset probabilities — one each for banking, currency, sovereign,
and fiscal crises. These four probabilities feed
``coordinator.CoordinatorRouter`` for joint inference, which in turn
drives ``analogy_engine.AnalogyEngine`` for retrieval and
``mitigation.MultiAgentMitigationPolicy`` for action selection.

The detectors are deliberately simple — one Conv1d-MLP per type, four
in total — so the locus of analogical reasoning is the downstream
``case_memory + analogy_engine`` substrate, not the raw signal
processing. This matches a long methodological tradition in the EM
crisis literature (Reinhart & Rogoff 2009; Frankel & Saravelos 2012):
crisis-onset signals are themselves not the hard problem; turning
detected crises into well-targeted policy is.

Architecture
------------
::

    pre_onset_trajectory (B, T_pre=20, F=12)
            │
            ▼
    BinaryCrisisDetector  -> per-type binary classifier with
                             Conv1d backbone + MLP head + sigmoid;
                             learned scalar temperature for
                             post-hoc calibration. Four instances,
                             one per type in DETECTOR_TYPES.
            │
            ▼
    DetectorEnsemble      -> orchestrator: holds the four detectors,
                             runs them in parallel, emits the 4-vector
                             that CoordinatorRouter consumes. Trains
                             all four detectors against one-vs-rest
                             binary labels derived from the case
                             library's crisis_type field, with
                             composite types (twin, triple) generating
                             positive labels for each of their
                             component detectors.
    DetectionOutput       -> immutable audit-trail dataclass holding
                             the 4 marginal probabilities, the
                             trajectory inputs, and per-detector
                             diagnostics. JSON-serializable.
    sample_negative_windows -> module-level helper that extracts
                             "no-crisis-imminent" sub-windows from
                             the early part of pre-onset trajectories,
                             producing the negative class for the
                             one-vs-rest binary training.

Composite-type positive labelling
---------------------------------
The coordinator vocabulary has composite types: ``twin`` (banking +
currency) and ``triple`` (banking + currency + sovereign). The
detection layer flattens these into per-detector binary labels:

* The ``banking`` detector treats every ``banking``, ``twin``, and
  ``triple`` case as positive.
* The ``currency`` detector treats every ``currency``, ``twin``, and
  ``triple`` case as positive.
* The ``sovereign`` detector treats every ``sovereign`` and ``triple``
  case as positive.
* The ``fiscal`` detector treats every ``sovereign`` and ``triple``
  case as positive — conceptually adjacent to sovereign but kept
  distinct so the coordinator can learn its slightly different
  feature signature (a fiscal crisis manifests in the budget balance
  and debt issuance levers before it shows up in spreads).

This mapping is captured in ``POSITIVE_TYPES_PER_DETECTOR`` and is the
single source of truth used by both training (label generation) and
inference (sanity-checking).

Reproducibility commitments
---------------------------
* All neural networks use seeded initialisation via save-and-restore
  of the global torch RNG.
* Mini-batch shuffling uses a single seeded CPU-side ``torch.Generator``.
* Calibration is via LBFGS, deterministic given a starting point.
* Negative-window sampling uses a seeded ``numpy.random.Generator``.
* All disk writes are atomic.

References (APA-7)
------------------
Frankel, J., & Saravelos, G. (2012). Can leading indicators assess
    country vulnerability? Evidence from the 2008-09 global financial
    crisis. Journal of International Economics, 87(2), 216-231.

Fukushima, K. (1980). Neocognitron: A self-organizing neural network
    model for a mechanism of pattern recognition unaffected by shift
    in position. Biological Cybernetics, 36(4), 193-202.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
    calibration of modern neural networks. In Proceedings of the
    34th International Conference on Machine Learning (Vol. 70,
    pp. 1321-1330).

Kaminsky, G. L., & Reinhart, C. M. (1999). The twin crises: The
    causes of banking and balance-of-payments problems. American
    Economic Review, 89(3), 473-500.

Laeven, L., & Valencia, F. (2018). Systemic banking crises revisited
    (IMF Working Paper No. 18/206). International Monetary Fund.

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature,
    521(7553), 436-444.

Reinhart, C. M., & Rogoff, K. S. (2009). This time is different:
    Eight centuries of financial folly. Princeton University Press.

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
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from case_memory import (
    CRISIS_TYPES,
    CaseLibrary,
    CaseMemoryError,
    DEFAULT_PRE_ONSET_QUARTERS,
    N_MACRO_FEATURES,
)
from coordinator import (
    DETECTOR_TYPES,
    N_DETECTOR_TYPES,
)


__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "POSITIVE_TYPES_PER_DETECTOR",
    "DEFAULT_CONV_CHANNELS",
    "DEFAULT_KERNEL_SIZE",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_DROPOUT",
    "DEFAULT_NEGATIVE_WINDOW_END",
    "DEFAULT_FIT_EPOCHS",
    # Exceptions
    "DetectionError",
    "DetectorNotFittedError",
    "InsufficientPositivesError",
    # Output dataclass
    "DetectionOutput",
    # Networks
    "BinaryCrisisDetector",
    "DetectorEnsemble",
    # Helpers
    "build_binary_labels",
    "sample_negative_windows",
    "extract_query_window",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Semantic version of the on-disk detection artefacts.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: Composite-type positive-label mapping.
POSITIVE_TYPES_PER_DETECTOR: Final[dict[str, tuple[str, ...]]] = {
    "banking":   ("banking", "twin", "triple"),
    "currency":  ("currency", "twin", "triple"),
    "sovereign": ("sovereign", "triple"),
    "fiscal":    ("sovereign", "triple"),
}

#: Pre-onset trajectory length in quarters.
T_PRE: Final[int] = DEFAULT_PRE_ONSET_QUARTERS

#: Macro-feature dimensionality.
N_FEATURES: Final[int] = N_MACRO_FEATURES

#: Default Conv1d channel widths in the backbone.
DEFAULT_CONV_CHANNELS: Final[tuple[int, int]] = (16, 32)

#: Default Conv1d kernel size.
DEFAULT_KERNEL_SIZE: Final[int] = 3

#: Default hidden dim of the MLP head.
DEFAULT_HIDDEN_DIM: Final[int] = 64

#: Default dropout.
DEFAULT_DROPOUT: Final[float] = 0.2

#: Default end-quarter for negative-window sampling.
DEFAULT_NEGATIVE_WINDOW_END: Final[int] = 8

#: Default training epochs.
DEFAULT_FIT_EPOCHS: Final[int] = 200

#: Default learning rate.
DEFAULT_LR: Final[float] = 1e-3

#: Default weight decay.
DEFAULT_WEIGHT_DECAY: Final[float] = 1e-4

#: Default batch size.
DEFAULT_BATCH_SIZE: Final[int] = 32

#: Default LBFGS iterations for temperature calibration.
DEFAULT_CALIBRATION_MAX_ITER: Final[int] = 100

#: Minimum positives required to train a detector.
_MIN_POSITIVES_FOR_FIT: Final[int] = 3

#: Atol for probability-range validation.
_PROB_ATOL: Final[float] = 1e-4


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DetectionError(CaseMemoryError):
    """Base class for detection-module exceptions."""


class DetectorNotFittedError(DetectionError):
    """An inference operation was called before ``fit``."""


class InsufficientPositivesError(DetectionError):
    """A detector's positive label set is too small to train reliably."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )


def _sha256_hex(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _check_compatible_version(
    found_version: str,
    expected_version: str = SCHEMA_VERSION,
    context: str = "object",
) -> None:
    try:
        found = tuple(int(p) for p in found_version.split("."))
        expected = tuple(int(p) for p in expected_version.split("."))
    except (ValueError, AttributeError) as exc:
        raise DetectionError(
            f"{context}: malformed schema_version {found_version!r}; "
            f"expected MAJOR.MINOR.PATCH"
        ) from exc
    if len(found) != 3 or len(expected) != 3:
        raise DetectionError(
            f"{context}: schema_version must be MAJOR.MINOR.PATCH; "
            f"got {found_version!r}"
        )
    if found[0] != expected[0]:
        raise DetectionError(
            f"{context}: schema major-version mismatch (found "
            f"{found_version!r}, code supports {expected_version!r})"
        )
    if found[1] != expected[1]:
        logger.warning(
            "%s: schema minor-version mismatch (found %s, code supports %s)",
            context, found_version, expected_version,
        )


def _seeded_module_init(seed: int, build_fn):
    saved_cpu = torch.get_rng_state()
    saved_cuda = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        return build_fn()
    finally:
        torch.set_rng_state(saved_cpu)
        if saved_cuda is not None:
            torch.cuda.set_rng_state_all(saved_cuda)


def _validate_trajectory(
    traj: np.ndarray, *, expected_T: int = T_PRE, expected_F: int = N_FEATURES,
    context: str = "trajectory",
) -> np.ndarray:
    # IMPORTANT: always return a fresh copy. Downstream
    # ``DetectionOutput.__post_init__`` sets ``query_trajectory`` read-only
    # to enforce audit-trail immutability; without an explicit copy here
    # that mutation would leak back to the caller's input array via
    # buffer sharing (``np.asarray`` does not copy when the dtype already
    # matches).
    arr = np.array(traj, dtype=np.float64, copy=True)
    if arr.ndim != 2:
        raise DetectionError(
            f"{context}: must be 2-D (T, F); got shape {arr.shape}"
        )
    if arr.shape[0] != expected_T:
        raise DetectionError(
            f"{context}: time axis {arr.shape[0]} != expected {expected_T}"
        )
    if arr.shape[1] != expected_F:
        raise DetectionError(
            f"{context}: feature axis {arr.shape[1]} != expected {expected_F}"
        )
    if np.any(~np.isfinite(arr)):
        raise DetectionError(f"{context}: contains non-finite entries")
    return arr


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_binary_labels(
    crisis_types: Sequence[str],
    detector_name: str,
) -> np.ndarray:
    """Construct binary one-vs-rest labels for one detector.

    Parameters
    ----------
    crisis_types : sequence of str
        Per-case crisis-type labels (each in ``case_memory.CRISIS_TYPES``).
    detector_name : str
        One of ``DETECTOR_TYPES``.

    Returns
    -------
    labels : np.ndarray of shape ``(N,)`` int64
    """
    if detector_name not in POSITIVE_TYPES_PER_DETECTOR:
        raise DetectionError(
            f"unknown detector {detector_name!r}; allowed "
            f"{list(POSITIVE_TYPES_PER_DETECTOR)}"
        )
    pos_set = POSITIVE_TYPES_PER_DETECTOR[detector_name]
    labels = np.zeros(len(crisis_types), dtype=np.int64)
    for i, ct in enumerate(crisis_types):
        if ct in pos_set:
            labels[i] = 1
        elif ct not in CRISIS_TYPES:
            raise DetectionError(
                f"build_binary_labels: case {i} has unknown crisis_type "
                f"{ct!r}; allowed {list(CRISIS_TYPES)}"
            )
    return labels


def sample_negative_windows(
    library: CaseLibrary,
    *,
    n_negatives: int,
    window_end: int = DEFAULT_NEGATIVE_WINDOW_END,
    seed: int = 42,
) -> np.ndarray:
    """Construct synthetic no-crisis-imminent trajectories.

    Each negative is built from one randomly-chosen library case by
    taking the first ``window_end`` quarters of its pre-onset
    trajectory (the "calm" period) and right-padding to ``T_PRE``
    quarters by repeating the last available quarter.

    Returns
    -------
    negatives : np.ndarray of shape ``(n_negatives, T_PRE, N_FEATURES)``
    """
    if len(library) == 0:
        raise DetectionError("sample_negative_windows: empty library")
    if n_negatives < 1:
        raise ValueError(f"n_negatives must be >= 1; got {n_negatives}")
    if window_end <= 0 or window_end > T_PRE:
        raise ValueError(
            f"window_end must be in (0, {T_PRE}]; got {window_end}"
        )

    rng = np.random.default_rng(int(seed))
    case_ids = library.case_ids()
    out = np.zeros((n_negatives, T_PRE, N_FEATURES), dtype=np.float64)
    for n in range(n_negatives):
        cid = case_ids[rng.integers(0, len(case_ids))]
        case = library[cid]
        calm = case.pre_onset_trajectory[:window_end]
        last_quarter = calm[-1]
        out[n, :window_end] = calm
        out[n, window_end:] = last_quarter[np.newaxis, :]
    return out


def extract_query_window(
    macro_trajectory: np.ndarray,
    *,
    asof_quarter: Optional[int] = None,
    window_length: int = T_PRE,
) -> np.ndarray:
    """Extract the most-recent ``window_length`` quarters ending at
    ``asof_quarter`` (inclusive) from a longer macro trajectory.
    """
    traj = np.asarray(macro_trajectory, dtype=np.float64)
    if traj.ndim != 2:
        raise DetectionError(
            f"macro_trajectory must be 2-D; got shape {traj.shape}"
        )
    if traj.shape[1] != N_FEATURES:
        raise DetectionError(
            f"macro_trajectory feature dim {traj.shape[1]} != {N_FEATURES}"
        )
    T_total = traj.shape[0]
    if asof_quarter is None:
        asof_quarter = T_total - 1
    if asof_quarter < 0 or asof_quarter >= T_total:
        raise DetectionError(
            f"asof_quarter {asof_quarter} out of range [0, {T_total})"
        )
    start = asof_quarter - window_length + 1
    if start < 0:
        raise DetectionError(
            f"insufficient history: asof_quarter={asof_quarter} requires "
            f"{window_length} prior quarters, but only {asof_quarter + 1} "
            f"are available"
        )
    return traj[start:asof_quarter + 1].copy()


# ---------------------------------------------------------------------------
# DetectionOutput
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionOutput:
    """Immutable record of one detector-ensemble call."""

    probabilities: np.ndarray
    logits: np.ndarray
    temperatures_used: np.ndarray
    query_trajectory: np.ndarray
    most_likely_detector: str

    def __post_init__(self) -> None:
        for name, arr, expected_shape in (
            ("probabilities", self.probabilities, (N_DETECTOR_TYPES,)),
            ("logits", self.logits, (N_DETECTOR_TYPES,)),
            ("temperatures_used", self.temperatures_used, (N_DETECTOR_TYPES,)),
        ):
            a = np.asarray(arr)
            if a.shape != expected_shape:
                raise DetectionError(
                    f"DetectionOutput.{name}: shape {a.shape} != "
                    f"{expected_shape}"
                )
        if self.query_trajectory.shape != (T_PRE, N_FEATURES):
            raise DetectionError(
                f"DetectionOutput.query_trajectory: shape "
                f"{self.query_trajectory.shape} != ({T_PRE}, {N_FEATURES})"
            )
        if np.any(self.probabilities < -_PROB_ATOL) or \
                np.any(self.probabilities > 1.0 + _PROB_ATOL):
            raise DetectionError(
                f"DetectionOutput.probabilities: out of [0, 1]; "
                f"min={float(self.probabilities.min())}, "
                f"max={float(self.probabilities.max())}"
            )
        if np.any(self.temperatures_used <= 0):
            raise DetectionError(
                f"DetectionOutput.temperatures_used: must all be > 0; "
                f"min={float(self.temperatures_used.min())}"
            )
        if not np.all(np.isfinite(self.logits)):
            raise DetectionError(
                "DetectionOutput.logits: contains non-finite entries"
            )
        if self.most_likely_detector not in DETECTOR_TYPES:
            raise DetectionError(
                f"DetectionOutput.most_likely_detector: unknown "
                f"{self.most_likely_detector!r}"
            )
        # Defensive: mark arrays read-only
        for arr in (self.probabilities, self.logits, self.temperatures_used,
                    self.query_trajectory):
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)

    @property
    def max_probability(self) -> float:
        return float(self.probabilities.max())

    @property
    def detector_disagreement(self) -> float:
        return float(np.std(self.probabilities))

    def as_probability_dict(self) -> dict[str, float]:
        return {
            name: float(self.probabilities[i])
            for i, name in enumerate(DETECTOR_TYPES)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "probabilities": [float(v) for v in self.probabilities],
            "logits": [float(v) for v in self.logits],
            "temperatures_used": [float(v) for v in self.temperatures_used],
            "detector_types": list(DETECTOR_TYPES),
            "most_likely_detector": str(self.most_likely_detector),
            "max_probability": float(self.max_probability),
            "detector_disagreement": float(self.detector_disagreement),
            "query_trajectory": self.query_trajectory.tolist(),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        if indent is None:
            return _canonical_json(self.to_dict())
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=indent,
            ensure_ascii=True, allow_nan=False,
        )

    def __repr__(self) -> str:
        probs = ", ".join(
            f"{name}={p:.3f}"
            for name, p in zip(DETECTOR_TYPES, self.probabilities)
        )
        return f"DetectionOutput({probs}, top={self.most_likely_detector!r})"


# ---------------------------------------------------------------------------
# BinaryCrisisDetector
# ---------------------------------------------------------------------------


class BinaryCrisisDetector(nn.Module):
    """Per-type binary crisis detector with Conv1d backbone + MLP head."""

    def __init__(
        self,
        detector_name: str,
        *,
        n_features: int = N_FEATURES,
        t_pre: int = T_PRE,
        conv_channels: tuple[int, int] = DEFAULT_CONV_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        init_temperature: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if detector_name not in DETECTOR_TYPES:
            raise DetectionError(
                f"detector_name {detector_name!r} not in {DETECTOR_TYPES}"
            )
        # conv_channels can come back from JSON as a list; accept both
        conv_channels = tuple(conv_channels)
        if len(conv_channels) != 2:
            raise ValueError(
                f"conv_channels must have length 2; got {conv_channels}"
            )
        for i, c in enumerate(conv_channels):
            if c < 1:
                raise ValueError(
                    f"conv_channels[{i}]: must be >= 1; got {c}"
                )
        if kernel_size < 1 or kernel_size > t_pre:
            raise ValueError(
                f"kernel_size must be in [1, t_pre={t_pre}]; "
                f"got {kernel_size}"
            )
        if hidden_dim < 4:
            raise ValueError(f"hidden_dim must be >= 4; got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")
        if init_temperature <= 0:
            raise ValueError(
                f"init_temperature must be > 0; got {init_temperature}"
            )

        self.detector_name: str = detector_name
        self.config: dict[str, Any] = {
            "detector_name": detector_name,
            "n_features": int(n_features),
            "t_pre": int(t_pre),
            "conv_channels": tuple(int(c) for c in conv_channels),
            "kernel_size": int(kernel_size),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "init_temperature": float(init_temperature),
            "seed": int(seed),
        }

        pad = kernel_size // 2

        def _build() -> nn.Module:
            return nn.Sequential(
                nn.Conv1d(n_features, conv_channels[0],
                          kernel_size=kernel_size, padding=pad),
                nn.GELU(),
                nn.Conv1d(conv_channels[0], conv_channels[1],
                          kernel_size=kernel_size, padding=pad),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(conv_channels[1], hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        self.net = _seeded_module_init(seed, _build)
        self.register_buffer(
            "log_temperature",
            torch.tensor(math.log(init_temperature), dtype=torch.float32),
        )
        self._is_fitted: bool = False
        self._fit_history: dict[str, list[float]] = {}

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Pre-sigmoid logit. Accepts (T, F) or (B, T, F)."""
        squeeze = False
        if trajectory.ndim == 2:
            trajectory = trajectory.unsqueeze(0)
            squeeze = True
        elif trajectory.ndim != 3:
            raise DetectionError(
                f"trajectory must be 2-D (T, F) or 3-D (B, T, F); "
                f"got shape {tuple(trajectory.shape)}"
            )
        if trajectory.shape[1] != self.config["t_pre"]:
            raise DetectionError(
                f"trajectory time dim {trajectory.shape[1]} != "
                f"t_pre {self.config['t_pre']}"
            )
        if trajectory.shape[2] != self.config["n_features"]:
            raise DetectionError(
                f"trajectory feature dim {trajectory.shape[2]} != "
                f"n_features {self.config['n_features']}"
            )
        x = trajectory.transpose(1, 2)  # (B, F, T)
        logit = self.net(x).squeeze(-1)
        return logit.squeeze(0) if squeeze else logit

    def probability(self, trajectory: torch.Tensor) -> torch.Tensor:
        logit = self.forward(trajectory)
        return torch.sigmoid(logit / self.temperature)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sd_path = path / "detector.pt"
        sd_tmp = sd_path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), sd_tmp)
        os.replace(sd_tmp, sd_path)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "config": self.config,
            "is_fitted": self._is_fitted,
            "current_temperature": float(self.temperature.item()),
        }
        _atomic_write_text(
            path / "detector_config.json",
            json.dumps(meta, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        return path

    @classmethod
    def load(
        cls, path: Path, *,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "BinaryCrisisDetector":
        path = Path(path)
        meta = json.loads(
            (path / "detector_config.json").read_text(encoding="utf-8"),
        )
        _check_compatible_version(
            meta.get("schema_version", SCHEMA_VERSION),
            context=f"BinaryCrisisDetector@{path}",
        )
        cfg = dict(meta["config"])
        cfg["conv_channels"] = tuple(cfg["conv_channels"])
        det = cls(**cfg)
        det.load_state_dict(
            torch.load(path / "detector.pt", map_location=map_location),
        )
        det._is_fitted = bool(meta.get("is_fitted", False))
        det.to(map_location)
        det.eval()
        return det

    def __repr__(self) -> str:
        return (
            f"BinaryCrisisDetector(name={self.detector_name!r}, "
            f"T={float(self.temperature.item()):.3f}, "
            f"fitted={self._is_fitted})"
        )


# ---------------------------------------------------------------------------
# DetectorEnsemble
# ---------------------------------------------------------------------------


class DetectorEnsemble(nn.Module):
    """Four binary crisis detectors, one per ``DETECTOR_TYPES``."""

    def __init__(
        self,
        *,
        n_features: int = N_FEATURES,
        t_pre: int = T_PRE,
        conv_channels: tuple[int, int] = DEFAULT_CONV_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        seed: int = 42,
    ) -> None:
        super().__init__()
        conv_channels = tuple(conv_channels)
        self.config: dict[str, Any] = {
            "n_features": int(n_features),
            "t_pre": int(t_pre),
            "conv_channels": tuple(int(c) for c in conv_channels),
            "kernel_size": int(kernel_size),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "seed": int(seed),
        }
        self.detectors = nn.ModuleDict({
            name: BinaryCrisisDetector(
                detector_name=name,
                n_features=n_features, t_pre=t_pre,
                conv_channels=conv_channels, kernel_size=kernel_size,
                hidden_dim=hidden_dim, dropout=dropout,
                seed=seed + i,
            )
            for i, name in enumerate(DETECTOR_TYPES)
        })
        self._is_fitted: bool = False
        self._fit_history: dict[str, dict[str, list[float]]] = {}

    def __getitem__(self, name: str) -> BinaryCrisisDetector:
        if name not in self.detectors:
            raise DetectionError(
                f"unknown detector {name!r}; allowed {list(self.detectors)}"
            )
        return self.detectors[name]

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Stack of pre-sigmoid logits from all four detectors."""
        squeeze = (trajectory.ndim == 2)
        if squeeze:
            trajectory = trajectory.unsqueeze(0)
        logits_list = [
            self.detectors[name](trajectory)
            for name in DETECTOR_TYPES
        ]
        out = torch.stack(logits_list, dim=-1)
        return out.squeeze(0) if squeeze else out

    def probabilities(self, trajectory: torch.Tensor) -> torch.Tensor:
        squeeze = (trajectory.ndim == 2)
        if squeeze:
            trajectory = trajectory.unsqueeze(0)
        probs_list = [
            self.detectors[name].probability(trajectory)
            for name in DETECTOR_TYPES
        ]
        out = torch.stack(probs_list, dim=-1)
        return out.squeeze(0) if squeeze else out

    def detect(self, trajectory: np.ndarray) -> DetectionOutput:
        """Single-trajectory inference -> DetectionOutput."""
        if not self._is_fitted:
            warnings.warn(
                "DetectorEnsemble.detect: ensemble has not been .fit. "
                "The probabilities are essentially random.",
                stacklevel=2,
            )
        traj_np = _validate_trajectory(trajectory, context="query trajectory")
        device = next(self.parameters()).device

        was_training = self.training
        self.eval()
        try:
            traj_t = torch.as_tensor(
                traj_np, dtype=torch.float32, device=device,
            )
            with torch.no_grad():
                logits = self.forward(traj_t)
                probs_list, temps_list = [], []
                for i, name in enumerate(DETECTOR_TYPES):
                    det = self.detectors[name]
                    probs_list.append(torch.sigmoid(logits[i] / det.temperature))
                    temps_list.append(det.temperature)
                probs_t = torch.stack(probs_list)
                temps_t = torch.stack(temps_list)
        finally:
            if was_training:
                self.train()

        probs_np = probs_t.cpu().numpy().astype(np.float64).clip(0.0, 1.0)
        logits_np = logits.cpu().numpy().astype(np.float64)
        temps_np = temps_t.cpu().numpy().astype(np.float64)
        argmax_idx = int(np.argmax(probs_np))

        return DetectionOutput(
            probabilities=probs_np,
            logits=logits_np,
            temperatures_used=temps_np,
            query_trajectory=traj_np,
            most_likely_detector=DETECTOR_TYPES[argmax_idx],
        )

    def fit(
        self,
        library: CaseLibrary,
        *,
        n_negatives_per_detector: Optional[int] = None,
        negative_window_end: int = DEFAULT_NEGATIVE_WINDOW_END,
        val_fraction: float = 0.15,
        n_epochs: int = DEFAULT_FIT_EPOCHS,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        pos_weight_balance: bool = True,
        grad_clip: float = 1.0,
        calibrate: bool = True,
        seed: int = 42,
        verbose: bool = False,
    ) -> dict[str, dict[str, list[float]]]:
        """Train all four detectors against one-vs-rest binary labels."""
        if len(library) == 0:
            raise DetectionError("DetectorEnsemble.fit: empty library")
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(
                f"val_fraction must be in [0, 1); got {val_fraction}"
            )

        case_ids = library.case_ids()
        crisis_types = [library[cid].crisis_type for cid in case_ids]
        n_pos_cases = len(case_ids)

        pos_trajs = np.stack([
            library[cid].pre_onset_trajectory for cid in case_ids
        ], axis=0).astype(np.float32)

        if n_negatives_per_detector is None:
            n_negatives_per_detector = n_pos_cases

        device = next(self.parameters()).device
        history: dict[str, dict[str, list[float]]] = {}
        rng_master = np.random.default_rng(int(seed))

        for det_idx, name in enumerate(DETECTOR_TYPES):
            pos_labels = build_binary_labels(crisis_types, name)
            pos_mask = pos_labels.astype(bool)
            n_pos = int(pos_mask.sum())
            if n_pos < _MIN_POSITIVES_FOR_FIT:
                raise InsufficientPositivesError(
                    f"detector {name!r}: only {n_pos} positives in library "
                    f"(need >= {_MIN_POSITIVES_FOR_FIT}). Positive types: "
                    f"{POSITIVE_TYPES_PER_DETECTOR[name]}"
                )
            X_pos = pos_trajs[pos_mask]
            y_pos = np.ones(n_pos, dtype=np.float32)

            det_seed = int(rng_master.integers(0, 2**31 - 1))
            X_neg = sample_negative_windows(
                library, n_negatives=n_negatives_per_detector,
                window_end=negative_window_end, seed=det_seed,
            ).astype(np.float32)
            y_neg = np.zeros(n_negatives_per_detector, dtype=np.float32)

            X = np.concatenate([X_pos, X_neg], axis=0)
            y = np.concatenate([y_pos, y_neg], axis=0)

            # Stratified train/val split
            if val_fraction > 0 and calibrate:
                pos_idx = np.where(y == 1)[0]
                neg_idx = np.where(y == 0)[0]
                rng_split = np.random.default_rng(det_seed + 1)
                rng_split.shuffle(pos_idx)
                rng_split.shuffle(neg_idx)
                n_pos_val = max(1, int(round(val_fraction * len(pos_idx))))
                n_neg_val = max(1, int(round(val_fraction * len(neg_idx))))
                val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
                train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                do_calibrate = True
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None
                do_calibrate = False

            X_train_t = torch.as_tensor(X_train, device=device)
            y_train_t = torch.as_tensor(y_train, device=device)

            if pos_weight_balance:
                n_pos_train = int(y_train.sum())
                n_neg_train = len(y_train) - n_pos_train
                if n_pos_train == 0:
                    pos_weight = torch.tensor(1.0, device=device)
                else:
                    pos_weight = torch.tensor(
                        max(1.0, n_neg_train / n_pos_train),
                        dtype=torch.float32, device=device,
                    )
            else:
                pos_weight = None

            det = self.detectors[name]
            optimizer = torch.optim.AdamW(
                det.net.parameters(), lr=lr, weight_decay=weight_decay,
            )
            gen = torch.Generator(device="cpu").manual_seed(det_seed + 2)

            det_hist: dict[str, list[float]] = {
                "epoch": [], "loss": [], "accuracy": [],
            }
            det.train()
            N_train = X_train_t.shape[0]
            for epoch in range(n_epochs):
                perm = torch.randperm(N_train, generator=gen).to(device)
                epoch_losses: list[float] = []
                for i in range(0, N_train, batch_size):
                    idx = perm[i:i + batch_size]
                    logit = det.forward(X_train_t[idx])
                    loss = F.binary_cross_entropy_with_logits(
                        logit, y_train_t[idx], pos_weight=pos_weight,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            det.net.parameters(), max_norm=grad_clip,
                        )
                    optimizer.step()
                    epoch_losses.append(float(loss.item()))

                det.eval()
                with torch.no_grad():
                    full_logit = det.forward(X_train_t)
                    pred = (torch.sigmoid(full_logit) >= 0.5).float()
                    acc = float((pred == y_train_t).float().mean().item())
                det.train()

                det_hist["epoch"].append(epoch)
                det_hist["loss"].append(float(np.mean(epoch_losses)))
                det_hist["accuracy"].append(acc)
                if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                    logger.info(
                        "DetectorEnsemble.fit[%s]: epoch %d/%d  "
                        "loss=%.5f  acc=%.4f",
                        name, epoch + 1, n_epochs, det_hist["loss"][-1], acc,
                    )

            det.eval()
            det._is_fitted = True
            det._fit_history = det_hist

            if do_calibrate and X_val is not None and y_val is not None:
                cal = self._calibrate_detector(
                    det,
                    torch.as_tensor(X_val, device=device),
                    torch.as_tensor(y_val, device=device),
                )
                det_hist["val_pre_nll"] = [float(cal["pre_nll"])]
                det_hist["val_post_nll"] = [float(cal["post_nll"])]
                det_hist["temperature"] = [float(cal["post_temperature"])]

            history[name] = det_hist
            if verbose:
                logger.info(
                    "DetectorEnsemble.fit[%s]: complete; final loss=%.4f, "
                    "acc=%.3f, T=%.3f",
                    name, det_hist["loss"][-1], det_hist["accuracy"][-1],
                    float(det.temperature.item()),
                )

        self._is_fitted = True
        self._fit_history = history
        return history

    def _calibrate_detector(
        self,
        det: BinaryCrisisDetector,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        *,
        max_iter: int = DEFAULT_CALIBRATION_MAX_ITER,
        lr: float = 1e-2,
    ) -> dict[str, float]:
        """Post-hoc temperature scaling for one detector."""
        det.eval()
        with torch.no_grad():
            val_logits = det.forward(X_val)
            pre_T = float(det.temperature.item())
            pre_nll = float(
                F.binary_cross_entropy_with_logits(
                    val_logits / pre_T, y_val,
                ).item()
            )

        log_T = det.log_temperature.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [log_T], lr=lr, max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            T = torch.exp(log_T)
            loss = F.binary_cross_entropy_with_logits(
                val_logits / T, y_val,
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        post_log_T = log_T.detach()
        if not torch.isfinite(post_log_T).all():
            warnings.warn(
                f"_calibrate_detector[{det.detector_name}]: LBFGS produced "
                f"non-finite log_temperature; reverting to pre-calibration.",
                stacklevel=2,
            )
            post_log_T = det.log_temperature.detach().clone()
        det.log_temperature.copy_(post_log_T.to(det.log_temperature.dtype))
        post_T = float(torch.exp(post_log_T).item())
        with torch.no_grad():
            post_nll = float(
                F.binary_cross_entropy_with_logits(
                    val_logits / torch.exp(post_log_T), y_val,
                ).item()
            )

        if post_nll > pre_nll + 1e-6:
            warnings.warn(
                f"_calibrate_detector[{det.detector_name}]: post_nll "
                f"{post_nll:.4f} > pre_nll {pre_nll:.4f}; LBFGS did not "
                f"converge to global optimum. Calibration applied anyway.",
                stacklevel=2,
            )

        return {
            "pre_temperature": pre_T,
            "post_temperature": post_T,
            "pre_nll": pre_nll,
            "post_nll": post_nll,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sd_path = path / "ensemble.pt"
        sd_tmp = sd_path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), sd_tmp)
        os.replace(sd_tmp, sd_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config": self.config,
            "is_fitted": self._is_fitted,
            "detector_names": list(DETECTOR_TYPES),
            "detector_temperatures": {
                name: float(self.detectors[name].temperature.item())
                for name in DETECTOR_TYPES
            },
            "fit_history_summary": {
                name: {
                    "n_epochs": len(self._fit_history.get(name, {}).get("loss", [])),
                    "final_loss": (
                        self._fit_history[name]["loss"][-1]
                        if name in self._fit_history
                        and self._fit_history[name].get("loss") else None
                    ),
                    "final_accuracy": (
                        self._fit_history[name]["accuracy"][-1]
                        if name in self._fit_history
                        and self._fit_history[name].get("accuracy") else None
                    ),
                    "calibration_temperature": (
                        self._fit_history[name]["temperature"][-1]
                        if name in self._fit_history
                        and self._fit_history[name].get("temperature") else None
                    ),
                }
                for name in DETECTOR_TYPES
            } if self._fit_history else {},
            "saved_at": _utc_now_iso(),
        }
        _atomic_write_text(
            path / "ensemble_manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "DetectorEnsemble":
        path = Path(path)
        manifest_path = path / "ensemble_manifest.json"
        sd_path = path / "ensemble.pt"
        if not manifest_path.is_file() or not sd_path.is_file():
            raise DetectionError(
                f"DetectorEnsemble.load: missing files under {path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _check_compatible_version(
            manifest.get("schema_version", SCHEMA_VERSION),
            context=f"DetectorEnsemble@{path}",
        )
        cfg = dict(manifest["config"])
        cfg["conv_channels"] = tuple(cfg["conv_channels"])
        ensemble = cls(**cfg)
        ensemble.load_state_dict(
            torch.load(sd_path, map_location=map_location),
        )
        ensemble._is_fitted = bool(manifest.get("is_fitted", False))
        if ensemble._is_fitted:
            for det in ensemble.detectors.values():
                det._is_fitted = True
        ensemble.to(map_location)
        ensemble.eval()
        return ensemble

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": dict(self.config),
            "n_detectors": len(self.detectors),
            "detector_names": list(self.detectors.keys()),
            "n_parameters_total": int(
                sum(p.numel() for p in self.parameters())
            ),
            "n_parameters_per_detector": {
                name: int(sum(p.numel() for p in det.parameters()))
                for name, det in self.detectors.items()
            },
            "current_temperatures": {
                name: float(self.detectors[name].temperature.item())
                for name in DETECTOR_TYPES
            },
            "is_fitted": self._is_fitted,
        }

    def __repr__(self) -> str:
        return (
            f"DetectorEnsemble(n_detectors={len(self.detectors)}, "
            f"fitted={self._is_fitted})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> int:
    """CLI: ``python -m em_fin_stability.detection <command> [args]``."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m em_fin_stability.detection",
        description="Inspect a saved DetectorEnsemble.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_sum = sub.add_parser("summary", help="Print ensemble summary.")
    p_sum.add_argument("ensemble_path", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _configure_logging(args.log_level)

    try:
        if args.command == "summary":
            ensemble = DetectorEnsemble.load(args.ensemble_path)
            print(json.dumps(ensemble.summary(), indent=2, sort_keys=True))
            return 0
        else:  # pragma: no cover
            parser.print_help()
            return 2
    except CaseMemoryError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
