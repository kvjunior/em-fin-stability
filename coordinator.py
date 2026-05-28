"""
coordinator.py
==============

Disagreement-aware router from per-type detector probabilities to a
unified crisis-type posterior.

This module sits between ``detection.py`` (which emits per-type crisis
probabilities) and the downstream consumers (``analogy_engine.py`` for
retrieval and ``mitigation.py`` for policy). Its job is the joint
inference step:

  * inputs:  4 marginal detector probabilities (banking, currency,
             sovereign, fiscal) plus the 12-dim current macro state;
  * output:  a 6-dim posterior over the joint crisis-type vocabulary
             ``("none", "banking", "currency", "sovereign", "twin",
             "triple")``, plus auxiliary disagreement metrics.

The 6-dim posterior is exactly what ``analogy_engine.
coordinator_posterior_to_crisis_posterior`` consumes to produce the
5-dim crisis-type posterior that conditions the retriever. The 1 -
``posterior[0]`` gives the scalar crisis probability used by the
training orchestrator to gate retrieval (no retrieval when the model
is confident no crisis is brewing).

Architecture
------------
::

    CoordinatorRouter        -> learned routing network. Takes
                                (detector_probs, macro_state) and emits
                                a calibrated softmax posterior over the
                                6 crisis-type categories.

                                Internal architecture: polynomial
                                expansion of the 4 detector probs
                                (15 features = all non-empty subsets of
                                {0,1,2,3}), concatenated with the
                                macro state, fed through a two-layer
                                MLP to a 6-class logit head. A learned
                                scalar temperature in log-space
                                produces calibrated posteriors via
                                Guo et al. 2017's post-hoc temperature
                                scaling.

                                ~3.8 k parameters for the canonical
                                configuration (input 27 -> hidden 64
                                -> hidden 64 -> 6).
    CoordinatorOutput        -> immutable record of one ``coordinate``
                                call. Carries the 6-dim posterior, the
                                crisis_probability scalar, two
                                disagreement metrics, the raw inputs
                                that produced it, and the temperature
                                in use. JSON-serializable for audit.

Why polynomial features
-----------------------
The semantic distinction between crisis-type labels is intrinsically
multiplicative: "twin" means banking and currency together; "triple"
means banking AND currency AND sovereign. A linear classifier over
the 4 detector outputs cannot represent these conjunctions. The
polynomial expansion makes every product term (banking*currency,
banking*currency*sovereign, ...) a first-class input, so the
routing decision is a linear function of these conjunctive features.

For n detectors the expansion has :math:`2^n - 1` features (all
non-empty subsets); for n = 4 this is 15, comfortably small for
a low-parameter classifier. The expansion is deterministic and
order-stable so the saved router's parameters retain their semantic
mapping across reloads.

Why temperature scaling
-----------------------
Cross-entropy training maximises log-likelihood but does not produce
calibrated probabilities — a model can be over- or under-confident
even with high accuracy. The downstream uses are highly sensitive to
calibration:

  * the scalar ``crisis_probability`` gates whether the analogy
    engine is invoked at all;
  * the 5-dim crisis-type posterior weights the per-type metrics in
    the conditional retriever;
  * the actor consumes the posterior as part of its context.

We follow Guo et al. (2017) and apply post-hoc temperature scaling:
after cross-entropy training, a single scalar T is fitted on a held-
out validation set by LBFGS minimisation of the NLL. Temperature does
not change accuracy (argmax is invariant to monotone transformations)
but markedly improves calibration.

Reproducibility commitments
---------------------------
* Network initialisation is deterministic via save-and-restore of the
  global torch RNG.
* The ``fit`` method's mini-batch shuffling uses a seeded torch
  Generator.
* Calibration uses LBFGS, which is deterministic given a starting
  point.
* All disk writes are atomic.

References (APA-7)
------------------
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
    calibration of modern neural networks. In Proceedings of the
    34th International Conference on Machine Learning (Vol. 70,
    pp. 1321-1330).

Laeven, L., & Valencia, F. (2018). Systemic banking crises revisited
    (IMF Working Paper No. 18/206). International Monetary Fund.

Reinhart, C. M., & Rogoff, K. S. (2009). This time is different:
    Eight centuries of financial folly. Princeton University Press.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good
    probabilities with supervised learning. In Proceedings of the
    22nd International Conference on Machine Learning (pp. 625-632).

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
from itertools import combinations
from pathlib import Path
from typing import Any, Final, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from case_memory import (
    CRISIS_TYPES,
    CaseMemoryError,
    N_MACRO_FEATURES,
)
from analogy_engine import (
    coordinator_posterior_to_crisis_posterior,
)


__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "COORDINATOR_TYPES",
    "DETECTOR_TYPES",
    "N_COORDINATOR_TYPES",
    "N_DETECTOR_TYPES",
    "POLY_FEATURE_DIM",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_DROPOUT",
    "DEFAULT_FIT_EPOCHS",
    # Exceptions
    "CoordinatorError",
    "RouterNotFittedError",
    # Outputs
    "CoordinatorOutput",
    # Network
    "CoordinatorRouter",
    # Helpers
    "detector_probs_to_polynomial_features",
    "polynomial_feature_dim",
    "polynomial_feature_names",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Semantic version of the on-disk coordinator artefact.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: Crisis-type vocabulary at the coordinator level: ``"none"`` at
#: index 0 followed by the 5 actual crisis types from
#: ``case_memory.CRISIS_TYPES``. The ordering is canonical and must
#: not be changed without bumping ``SCHEMA_VERSION`` since it is the
#: layout of the saved softmax head.
COORDINATOR_TYPES: Final[tuple[str, ...]] = ("none",) + CRISIS_TYPES

#: Detector vocabulary: the 4 per-type detectors whose probabilities
#: feed the router. ``fiscal`` is conceptually adjacent to
#: ``sovereign`` (it is the early-warning signal for sovereign
#: distress) but kept distinct so the router can learn their slightly
#: different signatures.
DETECTOR_TYPES: Final[tuple[str, ...]] = (
    "banking", "currency", "sovereign", "fiscal",
)

N_COORDINATOR_TYPES: Final[int] = len(COORDINATOR_TYPES)   # 6
N_DETECTOR_TYPES: Final[int] = len(DETECTOR_TYPES)         # 4

#: Polynomial feature dimensionality for the default 4-detector
#: configuration: :math:`2^4 - 1 = 15`. Indexed by all non-empty
#: subsets of ``{0, 1, 2, 3}`` in canonical (size-then-lex) order.
POLY_FEATURE_DIM: Final[int] = (1 << N_DETECTOR_TYPES) - 1

#: Macro-state dimensionality. Matches ``case_memory.N_MACRO_FEATURES``.
STATE_DIM: Final[int] = N_MACRO_FEATURES

#: Default hidden dim for the routing MLP. Small (64) because the
#: input dimensionality is small (27) and overfitting on the
#: ~150-case library is the chief concern.
DEFAULT_HIDDEN_DIM: Final[int] = 64

#: Default dropout in the routing MLP.
DEFAULT_DROPOUT: Final[float] = 0.1

#: Default training epochs.
DEFAULT_FIT_EPOCHS: Final[int] = 200

#: Default learning rate for the AdamW optimiser.
DEFAULT_LR: Final[float] = 1e-3

#: Default weight decay.
DEFAULT_WEIGHT_DECAY: Final[float] = 1e-4

#: Default mini-batch size.
DEFAULT_BATCH_SIZE: Final[int] = 32

#: Default label-smoothing strength for cross-entropy training. Mild
#: smoothing improves calibration even before temperature scaling.
DEFAULT_LABEL_SMOOTHING: Final[float] = 0.05

#: Default LBFGS iterations for temperature calibration.
DEFAULT_CALIBRATION_MAX_ITER: Final[int] = 100


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class CoordinatorError(CaseMemoryError):
    """Base class for coordinator-module exceptions. Inherits from
    ``CaseMemoryError`` so the package-wide ``except`` clause catches it."""


class RouterNotFittedError(CoordinatorError):
    """Raised when an inference path is exercised before ``fit``."""


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
        raise CoordinatorError(
            f"{context}: malformed schema_version {found_version!r}; "
            f"expected MAJOR.MINOR.PATCH"
        ) from exc
    if len(found) != 3 or len(expected) != 3:
        raise CoordinatorError(
            f"{context}: schema_version must be MAJOR.MINOR.PATCH; "
            f"got {found_version!r}"
        )
    if found[0] != expected[0]:
        raise CoordinatorError(
            f"{context}: schema major-version mismatch (found "
            f"{found_version!r}, code supports {expected_version!r})"
        )
    if found[1] != expected[1]:
        logger.warning(
            "%s: schema minor-version mismatch (found %s, code supports %s)",
            context, found_version, expected_version,
        )


def _seeded_module_init(seed: int, build_fn):
    """Build a torch module with deterministic init by saving and
    restoring the global RNG state around the build call."""
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


# ---------------------------------------------------------------------------
# Polynomial features
# ---------------------------------------------------------------------------


def _build_subset_indices(n: int) -> tuple[tuple[int, ...], ...]:
    """All non-empty subsets of ``{0, ..., n-1}`` in canonical
    (size-then-lex) order. Used for the polynomial-feature expansion."""
    if n < 1:
        raise CoordinatorError(f"_build_subset_indices: n must be >= 1; got {n}")
    out: list[tuple[int, ...]] = []
    for k in range(1, n + 1):
        for combo in combinations(range(n), k):
            out.append(combo)
    return tuple(out)


#: Cached subset indices for the default 4-detector configuration.
#: Order: 4 linear, then 6 pairwise, then 4 three-way, then 1 four-way.
_DEFAULT_SUBSETS: Final[tuple[tuple[int, ...], ...]] = _build_subset_indices(
    N_DETECTOR_TYPES,
)


def polynomial_feature_dim(n_detectors: int = N_DETECTOR_TYPES) -> int:
    """Polynomial feature dimensionality: ``2^n - 1`` for ``n`` detectors."""
    if n_detectors < 1:
        raise CoordinatorError(f"n_detectors must be >= 1; got {n_detectors}")
    return (1 << n_detectors) - 1


def polynomial_feature_names(
    detector_names: Sequence[str] = DETECTOR_TYPES,
) -> tuple[str, ...]:
    """Names of the polynomial features in canonical order.

    For detector_names = ('banking', 'currency', 'sovereign', 'fiscal')
    the returned tuple has 15 entries:

      banking, currency, sovereign, fiscal,
      banking*currency, banking*sovereign, banking*fiscal,
      currency*sovereign, currency*fiscal, sovereign*fiscal,
      banking*currency*sovereign, banking*currency*fiscal,
      banking*sovereign*fiscal, currency*sovereign*fiscal,
      banking*currency*sovereign*fiscal
    """
    names = tuple(detector_names)
    subsets = _build_subset_indices(len(names))
    return tuple(
        "*".join(names[i] for i in subset) for subset in subsets
    )


def detector_probs_to_polynomial_features(
    probs: torch.Tensor,
    n_detectors: int = N_DETECTOR_TYPES,
) -> torch.Tensor:
    """Map detector probabilities to polynomial features.

    Parameters
    ----------
    probs : Tensor of shape ``(..., n_detectors)``
        Per-detector marginal probabilities, each in ``[0, 1]`` (not
        validated here for differentiability; validation is done by
        the caller).
    n_detectors : int, default ``N_DETECTOR_TYPES``
        Number of detectors. Defaults match the package vocabulary.

    Returns
    -------
    features : Tensor of shape ``(..., 2^n_detectors - 1)``
        Polynomial features in canonical order (all non-empty subsets
        ordered by size then lex).

    Notes
    -----
    The expansion is differentiable: each output entry is a product of
    inputs and so has well-defined gradients. The default 4-detector
    case uses the cached subset list ``_DEFAULT_SUBSETS`` to avoid
    re-enumerating combinations on every call.
    """
    if probs.shape[-1] != n_detectors:
        raise CoordinatorError(
            f"polynomial features: last dim {probs.shape[-1]} != "
            f"n_detectors {n_detectors}"
        )
    if n_detectors == N_DETECTOR_TYPES:
        subsets = _DEFAULT_SUBSETS
    else:
        subsets = _build_subset_indices(n_detectors)
    features: list[torch.Tensor] = []
    for subset in subsets:
        feat = probs[..., subset[0]]
        for i in subset[1:]:
            feat = feat * probs[..., i]
        features.append(feat.unsqueeze(-1))
    return torch.cat(features, dim=-1)


def _validate_detector_probs(
    probs: np.ndarray,
    *,
    n_detectors: int = N_DETECTOR_TYPES,
    atol: float = 1e-4,
    context: str = "detector_probs",
) -> np.ndarray:
    """Validate detector outputs are in ``[0, 1]`` (within tolerance)
    and have the expected dim. Returns the array clipped to ``[0, 1]``."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim < 1:
        raise CoordinatorError(f"{context}: must be at least 1-D")
    if arr.shape[-1] != n_detectors:
        raise CoordinatorError(
            f"{context}: last dim {arr.shape[-1]} != n_detectors "
            f"{n_detectors}"
        )
    if np.any(~np.isfinite(arr)):
        raise CoordinatorError(f"{context}: contains non-finite entries")
    if np.any(arr < -atol) or np.any(arr > 1.0 + atol):
        raise CoordinatorError(
            f"{context}: out of [0, 1]; min={float(arr.min())}, "
            f"max={float(arr.max())}"
        )
    return arr.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# CoordinatorOutput dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoordinatorOutput:
    """Immutable record of one coordinator call.

    Carries everything needed for downstream consumers and for audit.
    A central-bank desk officer reading a policy recommendation sees
    this object's contents alongside the retrieval result: it tells
    them what the model believed about the kind of crisis brewing and
    how confident it was.

    Fields
    ------
    posterior : np.ndarray of shape ``(N_COORDINATOR_TYPES,)``
        Simplex posterior over ``COORDINATOR_TYPES``.
    crisis_probability : float
        ``1 - posterior[0]``, the marginal probability of any crisis.
        Used by the training orchestrator to gate retrieval.
    coordinator_entropy : float
        Shannon entropy of ``posterior`` in nats. Measures how
        uncertain the coordinator is about which class is correct.
    detector_disagreement : float
        Standard deviation of the 4 detector probabilities. Measures
        how consistent the upstream detector outputs were before
        coordination.
    detector_probs : np.ndarray of shape ``(N_DETECTOR_TYPES,)``
        The raw detector probabilities that produced this output, in
        ``DETECTOR_TYPES`` order, for audit.
    macro_state : np.ndarray of shape ``(STATE_DIM,)``
        The macro state that drove this output, for audit.
    most_likely_type : str
        ``COORDINATOR_TYPES[argmax(posterior)]``, the modal class.
    temperature_used : float
        The calibration temperature applied to the softmax. 1.0 means
        no scaling.
    """
    posterior: np.ndarray
    crisis_probability: float
    coordinator_entropy: float
    detector_disagreement: float
    detector_probs: np.ndarray
    macro_state: np.ndarray
    most_likely_type: str
    temperature_used: float

    def __post_init__(self) -> None:
        # Posterior shape and simplex check
        if self.posterior.shape != (N_COORDINATOR_TYPES,):
            raise CoordinatorError(
                f"CoordinatorOutput.posterior: shape {self.posterior.shape} "
                f"!= ({N_COORDINATOR_TYPES},)"
            )
        if np.any(self.posterior < -1e-6) or np.any(self.posterior > 1.0 + 1e-6):
            raise CoordinatorError(
                f"CoordinatorOutput.posterior: not on simplex "
                f"(min={float(self.posterior.min())}, "
                f"max={float(self.posterior.max())})"
            )
        if abs(float(self.posterior.sum()) - 1.0) > 1e-3:
            raise CoordinatorError(
                f"CoordinatorOutput.posterior: does not sum to 1 "
                f"(sum={float(self.posterior.sum())})"
            )
        # Detector probs
        if self.detector_probs.shape != (N_DETECTOR_TYPES,):
            raise CoordinatorError(
                f"CoordinatorOutput.detector_probs: shape "
                f"{self.detector_probs.shape} != ({N_DETECTOR_TYPES},)"
            )
        # Macro state
        if self.macro_state.shape != (STATE_DIM,):
            raise CoordinatorError(
                f"CoordinatorOutput.macro_state: shape "
                f"{self.macro_state.shape} != ({STATE_DIM},)"
            )
        # Most_likely_type sanity
        if self.most_likely_type not in COORDINATOR_TYPES:
            raise CoordinatorError(
                f"CoordinatorOutput.most_likely_type: unknown "
                f"{self.most_likely_type!r}"
            )
        # Crisis probability consistency
        expected_cp = 1.0 - float(self.posterior[0])
        if abs(float(self.crisis_probability) - expected_cp) > 1e-3:
            raise CoordinatorError(
                f"CoordinatorOutput.crisis_probability: "
                f"{self.crisis_probability:.4f} inconsistent with "
                f"1 - posterior[0] = {expected_cp:.4f}"
            )
        if not np.isfinite(self.coordinator_entropy):
            raise CoordinatorError(
                "CoordinatorOutput.coordinator_entropy: not finite"
            )
        if not np.isfinite(self.detector_disagreement):
            raise CoordinatorError(
                "CoordinatorOutput.detector_disagreement: not finite"
            )
        if not np.isfinite(self.temperature_used) or self.temperature_used <= 0:
            raise CoordinatorError(
                f"CoordinatorOutput.temperature_used: must be positive "
                f"finite; got {self.temperature_used}"
            )
        # Defensive: mark arrays read-only
        for arr in (self.posterior, self.detector_probs, self.macro_state):
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)

    # ------------------------------------------------------------------ #
    # Derived accessors
    # ------------------------------------------------------------------ #

    @property
    def crisis_type_posterior(self) -> np.ndarray:
        """The 5-dim posterior over ``CRISIS_TYPES`` (no ``"none"``),
        renormalised. This is the array that ``AnalogyEngine.retrieve``
        expects as ``type_posterior``."""
        return coordinator_posterior_to_crisis_posterior(self.posterior)

    @property
    def is_crisis_likely(self) -> bool:
        """Convenience flag: ``crisis_probability >= 0.5``. Callers
        may gate retrieval and policy on this or any other threshold."""
        return float(self.crisis_probability) >= 0.5

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation. Numpy arrays become lists."""
        return {
            "posterior": [float(v) for v in self.posterior],
            "coordinator_types": list(COORDINATOR_TYPES),
            "crisis_probability": float(self.crisis_probability),
            "coordinator_entropy": float(self.coordinator_entropy),
            "detector_disagreement": float(self.detector_disagreement),
            "detector_probs": [float(v) for v in self.detector_probs],
            "detector_types": list(DETECTOR_TYPES),
            "macro_state": [float(v) for v in self.macro_state],
            "most_likely_type": str(self.most_likely_type),
            "temperature_used": float(self.temperature_used),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        if indent is None:
            return _canonical_json(self.to_dict())
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=indent,
            ensure_ascii=True, allow_nan=False,
        )

    def __repr__(self) -> str:
        return (
            f"CoordinatorOutput(most_likely={self.most_likely_type!r}, "
            f"crisis_p={self.crisis_probability:.3f}, "
            f"entropy={self.coordinator_entropy:.3f})"
        )


# ---------------------------------------------------------------------------
# CoordinatorRouter
# ---------------------------------------------------------------------------


class CoordinatorRouter(nn.Module):
    """Disagreement-aware router from detector probabilities to a
    calibrated crisis-type posterior.

    Architecture
    ------------
    ::

        detector_probs (B, 4) --polynomial expansion--> (B, 15)
                                                              \\
                                                               concat --> (B, 27)
                                                              /
        macro_state    (B, 12) -----------------------------/
                                                              |
                                                              v
                                              Linear(27, 64) + GELU + Dropout
                                                              v
                                              Linear(64, 64) + GELU + Dropout
                                                              v
                                              Linear(64, 6)   [logits]
                                                              v
                                              softmax(logits / T)   [posterior]

    where ``T`` is a learned calibration temperature in log-space
    (parameterised as ``log_temperature``).

    Training
    --------
    ``fit`` trains the MLP on cross-entropy loss with mild label
    smoothing. The temperature is held at 1.0 during training and
    calibrated post-hoc by ``calibrate``, which fits ``T`` via
    LBFGS on a held-out validation set. ``fit`` calls ``calibrate``
    automatically if a validation set is supplied.

    Parameter count
    ---------------
    For the canonical configuration (poly_dim=15, state_dim=12,
    hidden_dim=64, n_classes=6): ``Linear(27, 64)`` = 1,792 +
    ``Linear(64, 64)`` = 4,160 + ``Linear(64, 6)`` = 390 = 6,342 total
    parameters in the MLP, plus the scalar ``log_temperature`` buffer.

    Note: ``log_temperature`` is a *buffer*, not a parameter, during
    cross-entropy training so the AdamW optimiser does not adjust it.
    Calibration uses a separate LBFGS optimiser over a detached copy
    of ``log_temperature`` and writes the result back to the buffer.
    """

    def __init__(
        self,
        *,
        macro_state_dim: int = STATE_DIM,
        n_classes: int = N_COORDINATOR_TYPES,
        n_detectors: int = N_DETECTOR_TYPES,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        init_temperature: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if macro_state_dim < 1:
            raise ValueError(
                f"macro_state_dim must be >= 1; got {macro_state_dim}"
            )
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2; got {n_classes}")
        if n_detectors < 1:
            raise ValueError(f"n_detectors must be >= 1; got {n_detectors}")
        if hidden_dim < 4:
            raise ValueError(f"hidden_dim must be >= 4; got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")
        if init_temperature <= 0:
            raise ValueError(
                f"init_temperature must be > 0; got {init_temperature}"
            )

        self.config: dict[str, Any] = {
            "macro_state_dim": int(macro_state_dim),
            "n_classes": int(n_classes),
            "n_detectors": int(n_detectors),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "init_temperature": float(init_temperature),
            "seed": int(seed),
        }
        self.poly_dim: int = polynomial_feature_dim(n_detectors)
        input_dim = self.poly_dim + macro_state_dim

        def _build() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        self.net = _seeded_module_init(seed, _build)
        # Temperature held as a buffer so the main optimiser doesn't
        # touch it. Calibration uses a separate path.
        self.register_buffer(
            "log_temperature",
            torch.tensor(math.log(init_temperature), dtype=torch.float32),
        )
        self._is_fitted: bool = False
        self._fit_history: dict[str, list[float]] = {}

    @property
    def temperature(self) -> torch.Tensor:
        """Scalar tensor: ``exp(log_temperature)``."""
        return torch.exp(self.log_temperature)

    # ------------------------------------------------------------------ #
    # Forward and inference
    # ------------------------------------------------------------------ #

    def forward(
        self,
        detector_probs: torch.Tensor,
        macro_state: torch.Tensor,
    ) -> torch.Tensor:
        """Raw logits (before temperature).

        Parameters
        ----------
        detector_probs : Tensor of shape ``(B, n_detectors)``
        macro_state : Tensor of shape ``(B, macro_state_dim)``

        Returns
        -------
        logits : Tensor of shape ``(B, n_classes)``
        """
        if detector_probs.ndim != 2:
            raise CoordinatorError(
                f"detector_probs must be 2-D; got shape "
                f"{tuple(detector_probs.shape)}"
            )
        if macro_state.ndim != 2:
            raise CoordinatorError(
                f"macro_state must be 2-D; got shape "
                f"{tuple(macro_state.shape)}"
            )
        if detector_probs.shape[0] != macro_state.shape[0]:
            raise CoordinatorError(
                f"batch mismatch: detector_probs has B={detector_probs.shape[0]}, "
                f"macro_state has B={macro_state.shape[0]}"
            )
        if detector_probs.shape[-1] != self.config["n_detectors"]:
            raise CoordinatorError(
                f"detector_probs last dim {detector_probs.shape[-1]} != "
                f"n_detectors {self.config['n_detectors']}"
            )
        if macro_state.shape[-1] != self.config["macro_state_dim"]:
            raise CoordinatorError(
                f"macro_state last dim {macro_state.shape[-1]} != "
                f"macro_state_dim {self.config['macro_state_dim']}"
            )
        poly = detector_probs_to_polynomial_features(
            detector_probs, n_detectors=self.config["n_detectors"],
        )
        z = torch.cat([poly, macro_state], dim=-1)
        return self.net(z)

    def posterior(
        self,
        detector_probs: torch.Tensor,
        macro_state: torch.Tensor,
    ) -> torch.Tensor:
        """Calibrated softmax posterior: ``softmax(logits / T)``."""
        logits = self.forward(detector_probs, macro_state)
        return F.softmax(logits / self.temperature, dim=-1)

    def coordinate(
        self,
        macro_state: np.ndarray,
        detector_probs: np.ndarray,
    ) -> CoordinatorOutput:
        """Single-state inference producing a ``CoordinatorOutput``.

        Parameters
        ----------
        macro_state : np.ndarray of shape ``(STATE_DIM,)``
            Current macro state.
        detector_probs : np.ndarray of shape ``(N_DETECTOR_TYPES,)``
            Per-detector probabilities, each in ``[0, 1]``.

        Returns
        -------
        CoordinatorOutput
        """
        if not self._is_fitted:
            warnings.warn(
                "CoordinatorRouter.coordinate: router has not been .fit. "
                "The posterior is essentially random.",
                stacklevel=2,
            )
        # Validate
        ms_arr = np.asarray(macro_state, dtype=np.float64)
        if ms_arr.shape != (self.config["macro_state_dim"],):
            raise CoordinatorError(
                f"macro_state shape {ms_arr.shape} != "
                f"({self.config['macro_state_dim']},)"
            )
        if np.any(~np.isfinite(ms_arr)):
            raise CoordinatorError("macro_state has non-finite entries")
        dp_arr = _validate_detector_probs(
            detector_probs, n_detectors=self.config["n_detectors"],
        )
        # Inference: temporarily put module in eval to disable dropout,
        # then restore.
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            dp_t = torch.as_tensor(
                dp_arr.copy(), dtype=torch.float32, device=device,
            ).unsqueeze(0)
            ms_t = torch.as_tensor(
                ms_arr.copy(), dtype=torch.float32, device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                post_t = self.posterior(dp_t, ms_t)[0]
        finally:
            if was_training:
                self.train()

        post_np = post_t.cpu().numpy().astype(np.float64)
        # Numerical: ensure simplex up to tiny float32 -> float64 drift
        post_np = post_np.clip(0.0, 1.0)
        post_np = post_np / post_np.sum()

        # Entropy (in nats)
        eps = 1e-12
        entropy = float(-np.sum(post_np * np.log(post_np + eps)))
        # Detector disagreement: std of detector probs
        disagreement = float(np.std(dp_arr))
        # Most likely
        argmax_idx = int(np.argmax(post_np))

        return CoordinatorOutput(
            posterior=post_np,
            crisis_probability=float(1.0 - post_np[0]),
            coordinator_entropy=entropy,
            detector_disagreement=disagreement,
            detector_probs=dp_arr.copy(),
            macro_state=ms_arr.copy(),
            most_likely_type=COORDINATOR_TYPES[argmax_idx],
            temperature_used=float(self.temperature.item()),
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(
        self,
        detector_probs: np.ndarray,
        macro_states: np.ndarray,
        labels: np.ndarray,
        *,
        val_detector_probs: Optional[np.ndarray] = None,
        val_macro_states: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        n_epochs: int = DEFAULT_FIT_EPOCHS,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
        grad_clip: float = 1.0,
        seed: int = 42,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train the router via cross-entropy.

        Parameters
        ----------
        detector_probs : np.ndarray of shape ``(N, n_detectors)``
            Training detector outputs, each row in ``[0, 1]``.
        macro_states : np.ndarray of shape ``(N, macro_state_dim)``
            Training macro states.
        labels : np.ndarray of shape ``(N,)`` integer
            Labels in ``{0, ..., n_classes-1}`` indexing
            ``COORDINATOR_TYPES``.
        val_detector_probs, val_macro_states, val_labels : optional
            Validation set. If all three are provided, the router is
            temperature-calibrated against them at the end of training.
        n_epochs, lr, weight_decay, batch_size, label_smoothing, grad_clip, seed
            Optimisation hyperparameters.
        verbose : bool
            Log epoch-level progress at INFO level.

        Returns
        -------
        history : dict
            ``"epoch"``, ``"loss"``, ``"accuracy"``, and if
            calibration ran, ``"val_pre_nll"``, ``"val_post_nll"``,
            ``"temperature"`` keys.
        """
        # ----- Input validation -----
        dp = np.asarray(detector_probs, dtype=np.float32)
        ms = np.asarray(macro_states, dtype=np.float32)
        lb = np.asarray(labels, dtype=np.int64)

        N = dp.shape[0]
        if N == 0:
            raise CoordinatorError("fit: empty dataset")
        if ms.shape[0] != N or lb.shape[0] != N:
            raise CoordinatorError(
                f"fit: row-count mismatch "
                f"(detector_probs={N}, macro_states={ms.shape[0]}, "
                f"labels={lb.shape[0]})"
            )
        if dp.shape[1] != self.config["n_detectors"]:
            raise CoordinatorError(
                f"fit: detector_probs cols {dp.shape[1]} != "
                f"n_detectors {self.config['n_detectors']}"
            )
        if ms.shape[1] != self.config["macro_state_dim"]:
            raise CoordinatorError(
                f"fit: macro_states cols {ms.shape[1]} != "
                f"macro_state_dim {self.config['macro_state_dim']}"
            )
        if np.any((lb < 0) | (lb >= self.config["n_classes"])):
            raise CoordinatorError(
                f"fit: labels must be in [0, {self.config['n_classes']});  "
                f"min={int(lb.min())}, max={int(lb.max())}"
            )
        if np.any(~np.isfinite(dp)) or np.any(~np.isfinite(ms)):
            raise CoordinatorError("fit: non-finite values in inputs")
        if np.any((dp < -1e-4) | (dp > 1.0 + 1e-4)):
            raise CoordinatorError(
                f"fit: detector_probs outside [0, 1] "
                f"(min={float(dp.min())}, max={float(dp.max())})"
            )
        dp = dp.clip(0.0, 1.0)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1); got {label_smoothing}"
            )

        # Validation set (optional)
        do_calibrate = (
            val_detector_probs is not None
            and val_macro_states is not None
            and val_labels is not None
        )
        if do_calibrate:
            vdp = np.asarray(val_detector_probs, dtype=np.float32)
            vms = np.asarray(val_macro_states, dtype=np.float32)
            vlb = np.asarray(val_labels, dtype=np.int64)
            if vdp.shape[1] != self.config["n_detectors"]:
                raise CoordinatorError(
                    "fit: val_detector_probs has wrong cols"
                )
            if vms.shape[1] != self.config["macro_state_dim"]:
                raise CoordinatorError(
                    "fit: val_macro_states has wrong cols"
                )
            vdp = vdp.clip(0.0, 1.0)

        # ----- Training -----
        device = next(self.parameters()).device
        dp_t = torch.as_tensor(dp, device=device)
        ms_t = torch.as_tensor(ms, device=device)
        lb_t = torch.as_tensor(lb, device=device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=lr, weight_decay=weight_decay,
        )

        # Mini-batch shuffling uses a CPU-side seeded Generator.
        gen = torch.Generator(device="cpu").manual_seed(int(seed))

        history: dict[str, list[float]] = {
            "epoch": [], "loss": [], "accuracy": [],
        }
        self.train()
        for epoch in range(n_epochs):
            perm = torch.randperm(N, generator=gen).to(device)
            epoch_losses: list[float] = []
            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                logits = self.forward(dp_t[idx], ms_t[idx])
                loss = F.cross_entropy(
                    logits, lb_t[idx], label_smoothing=label_smoothing,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), max_norm=grad_clip,
                    )
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            # End-of-epoch accuracy on full training set
            self.eval()
            with torch.no_grad():
                full_logits = self.forward(dp_t, ms_t)
                acc = float((full_logits.argmax(dim=-1) == lb_t).float().mean().item())
            self.train()

            history["epoch"].append(epoch)
            history["loss"].append(float(np.mean(epoch_losses)))
            history["accuracy"].append(acc)
            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                logger.info(
                    "CoordinatorRouter.fit: epoch %d/%d  loss=%.5f  acc=%.4f",
                    epoch + 1, n_epochs, history["loss"][-1], acc,
                )

        self.eval()
        self._is_fitted = True
        self._fit_history = history

        # ----- Optional temperature calibration -----
        if do_calibrate:
            cal = self.calibrate(vdp, vms, vlb)
            history["val_pre_nll"] = [float(cal["pre_nll"])]
            history["val_post_nll"] = [float(cal["post_nll"])]
            history["temperature"] = [float(cal["post_temperature"])]
        return history

    # ------------------------------------------------------------------ #
    # Calibration (temperature scaling)
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        val_detector_probs: np.ndarray,
        val_macro_states: np.ndarray,
        val_labels: np.ndarray,
        *,
        max_iter: int = DEFAULT_CALIBRATION_MAX_ITER,
        lr: float = 1e-2,
    ) -> dict[str, float]:
        """Post-hoc temperature scaling (Guo et al. 2017).

        Fits a single scalar temperature ``T`` to minimise the NLL on
        the supplied validation set. Argmax accuracy is invariant
        under temperature scaling, so this only affects calibration,
        not the predicted class.

        Returns a summary dict: ``"pre_temperature"``,
        ``"post_temperature"``, ``"pre_nll"``, ``"post_nll"``.
        """
        if not self._is_fitted:
            warnings.warn(
                "CoordinatorRouter.calibrate: router has not been .fit; "
                "calibration of an untrained model is meaningless but "
                "will proceed for protocol completeness.",
                stacklevel=2,
            )
        # Validate inputs
        vdp = _validate_detector_probs(
            val_detector_probs, n_detectors=self.config["n_detectors"],
            context="val_detector_probs",
        ).astype(np.float32)
        vms = np.asarray(val_macro_states, dtype=np.float32)
        if vms.shape[-1] != self.config["macro_state_dim"]:
            raise CoordinatorError(
                f"calibrate: val_macro_states last dim "
                f"{vms.shape[-1]} != macro_state_dim "
                f"{self.config['macro_state_dim']}"
            )
        vlb = np.asarray(val_labels, dtype=np.int64)
        if vlb.shape != (vdp.shape[0],):
            raise CoordinatorError(
                f"calibrate: val_labels shape {vlb.shape} != "
                f"({vdp.shape[0]},)"
            )
        if np.any((vlb < 0) | (vlb >= self.config["n_classes"])):
            raise CoordinatorError(
                "calibrate: val_labels out of range"
            )

        device = next(self.parameters()).device
        vdp_t = torch.as_tensor(vdp, device=device)
        vms_t = torch.as_tensor(vms, device=device)
        vlb_t = torch.as_tensor(vlb, device=device)

        # Compute logits once (no gradient needed through the net)
        self.eval()
        with torch.no_grad():
            val_logits = self.forward(vdp_t, vms_t)
            pre_T = float(self.temperature.item())
            pre_nll = float(F.cross_entropy(val_logits / pre_T, vlb_t).item())

        # Optimise log_T only; keep all network params frozen.
        log_T = self.log_temperature.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [log_T], lr=lr, max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            T = torch.exp(log_T)
            loss = F.cross_entropy(val_logits / T, vlb_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        post_log_T = log_T.detach()
        # Guard against pathological optimiser outcomes (NaN, exploding)
        if not torch.isfinite(post_log_T).all():
            warnings.warn(
                "CoordinatorRouter.calibrate: LBFGS produced non-finite "
                "log_temperature; reverting to pre-calibration value.",
                stacklevel=2,
            )
            post_log_T = self.log_temperature.detach().clone()

        # Write back
        self.log_temperature.copy_(post_log_T.to(self.log_temperature.dtype))
        post_T = float(torch.exp(post_log_T).item())
        with torch.no_grad():
            post_nll = float(
                F.cross_entropy(val_logits / torch.exp(post_log_T), vlb_t).item()
            )

        if post_nll > pre_nll + 1e-6:
            # Should not happen; LBFGS is monotone for convex objectives.
            warnings.warn(
                f"CoordinatorRouter.calibrate: post_nll {post_nll:.4f} > "
                f"pre_nll {pre_nll:.4f}; LBFGS did not converge to a "
                f"global optimum. Calibration applied anyway.",
                stacklevel=2,
            )

        logger.info(
            "CoordinatorRouter.calibrate: T %.4f -> %.4f, NLL %.4f -> %.4f",
            pre_T, post_T, pre_nll, post_nll,
        )
        return {
            "pre_temperature": pre_T,
            "post_temperature": post_T,
            "pre_nll": pre_nll,
            "post_nll": post_nll,
        }

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sd_path = path / "router.pt"
        sd_tmp = sd_path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), sd_tmp)
        os.replace(sd_tmp, sd_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config": self.config,
            "is_fitted": self._is_fitted,
            "current_temperature": float(self.temperature.item()),
            "fit_history_summary": (
                {
                    "n_epochs": len(self._fit_history.get("loss", [])),
                    "final_loss": (
                        self._fit_history["loss"][-1]
                        if self._fit_history.get("loss") else None
                    ),
                    "final_accuracy": (
                        self._fit_history["accuracy"][-1]
                        if self._fit_history.get("accuracy") else None
                    ),
                    "post_calibration_temperature": (
                        self._fit_history["temperature"][-1]
                        if self._fit_history.get("temperature") else None
                    ),
                }
                if self._fit_history else {}
            ),
            "saved_at": _utc_now_iso(),
        }
        _atomic_write_text(
            path / "router_config.json",
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
    ) -> "CoordinatorRouter":
        path = Path(path)
        meta_path = path / "router_config.json"
        sd_path = path / "router.pt"
        if not meta_path.is_file() or not sd_path.is_file():
            raise CoordinatorError(
                f"CoordinatorRouter.load: missing files under {path}"
            )
        manifest = json.loads(meta_path.read_text(encoding="utf-8"))
        _check_compatible_version(
            manifest.get("schema_version", SCHEMA_VERSION),
            context=f"CoordinatorRouter@{path}",
        )
        router = cls(**manifest["config"])
        router.load_state_dict(
            torch.load(sd_path, map_location=map_location),
        )
        router._is_fitted = bool(manifest.get("is_fitted", False))
        router.to(map_location)
        router.eval()
        return router

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": dict(self.config),
            "poly_dim": self.poly_dim,
            "n_parameters": int(sum(p.numel() for p in self.parameters())),
            "is_fitted": self._is_fitted,
            "current_temperature": float(self.temperature.item()),
            "fit_history_n_epochs": len(self._fit_history.get("loss", [])),
            "final_loss": (
                self._fit_history["loss"][-1]
                if self._fit_history.get("loss") else None
            ),
            "final_accuracy": (
                self._fit_history["accuracy"][-1]
                if self._fit_history.get("accuracy") else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"CoordinatorRouter(n_detectors={self.config['n_detectors']}, "
            f"n_classes={self.config['n_classes']}, "
            f"hidden_dim={self.config['hidden_dim']}, "
            f"T={float(self.temperature.item()):.3f}, "
            f"fitted={self._is_fitted})"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> int:
    """CLI: inspect a saved coordinator router.

    Commands
    --------
    summary <router_path>
        Load the router and print a JSON summary.
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m em_fin_stability.coordinator",
        description="Inspect a saved CoordinatorRouter.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_sum = sub.add_parser("summary", help="Print router summary.")
    p_sum.add_argument("router_path", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _configure_logging(args.log_level)

    try:
        if args.command == "summary":
            router = CoordinatorRouter.load(args.router_path)
            print(json.dumps(router.summary(), indent=2, sort_keys=True))
            return 0
        else:  # pragma: no cover
            parser.print_help()
            return 2
    except CaseMemoryError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
