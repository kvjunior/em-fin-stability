"""
analogy_engine.py
=================

Crisis-type-conditional retrieval and case-context encoding over a Crisis
Case Memory (novelty #2 of the paper).

This module is the reasoning counterpart to ``case_memory.py``. Where
``case_memory.py`` defines what a case *is* (a structured, immutable,
auditable knowledge record), ``analogy_engine.py`` defines how the model
*reasons over* a library of cases — specifically, how it selects the
historical analogues that are most informative for a current crisis-
risk surface, and how it summarises the selected analogues into a
fixed-size context vector that the multi-agent mitigation policy
consumes alongside the live macro state.

The core methodological contribution lives in the
``ConditionalRetriever``: a Mahalanobis-style similarity metric whose
positive-(semi-)definite matrix is conditioned on the inferred crisis
type. Standard retrieval-augmented systems use a *fixed* metric (cosine
similarity, dot product, or a single learned bilinear form). The
financial-stability domain demands more: when the coordinator believes
the brewing crisis is a *currency* episode, retrieval should emphasise
FX-reserve and REER signatures; when it believes a *sovereign* episode,
retrieval should emphasise spread and external-debt signatures. The
conditional metric formalises this domain intuition as a learnable,
end-to-end-trainable component.

Architecture
------------
::

    ConditionalRetriever     -> learns a mixture-of-Mahalanobis metric
                                indexed by crisis type:
                                   s(x, c | π) = Σ_τ π_τ · x^T W_τ c
                                with W_τ = L_τ L_τ^T + γ I (PSD by
                                construction via Cholesky factorisation).
                                Top-K selection via softmax-temperature
                                with both soft (training, gradient) and
                                hard (inference, audit) modes.
    CaseContextEncoder       -> per-case deterministic feature
                                construction (signature + 24-dim policy
                                fingerprint + 2-dim outcome fingerprint),
                                attention-pooled over retrieved cases
                                with retrieval similarity weights, then
                                projected to a context_dim-dim vector.
    AnalogyEngine            -> orchestrator: maintains the precomputed
                                signature / policy / outcome indices,
                                runs retrieve(), runs fit(), persists
                                state, and verifies library-checksum
                                consistency on load.
    RetrievalResult          -> immutable record of one retrieval call;
                                contains retrieved case_ids, raw
                                similarities, softmax weights, the
                                conditioning type posterior, the query
                                signature, and the aggregated context
                                vector. Suitable for storage as an
                                audit trail alongside each policy
                                recommendation.

Why this matters
----------------
The paper's claim is that the analogical-reasoning substrate is what
turns a black-box ML crisis-mitigation system into a deployable
decision-support tool. Every policy recommendation produced by the
downstream MARL actor is grounded in a retrieved set of historical
analogues; the AnalogyEngine produces both the context vector that
informs the actor and the human-readable case-id list that justifies
the recommendation to a central-bank desk officer. This is the
KBS-aesthetic property: reasoning is explicit, retrieved evidence is
inspectable, the knowledge artefact is queryable.

Reproducibility commitments
---------------------------
* Every parameter init is deterministic: seeded via the global torch
  RNG with save/restore so the caller's RNG state is preserved.
* The training procedure uses a single seeded ``torch.Generator`` for
  any sampling (currently full-batch only, but extensible).
* The engine records the library's content checksum at save time;
  ``load`` accepts a library and either verifies the checksum matches
  (strict, default) or warns and proceeds (lenient mode for
  development).
* All disk writes are atomic (``tmp`` + ``os.replace``).
* The case-feature standardisation statistics (policy and outcome
  means / stds) are computed from the *training* library exactly
  once at fit time and stored as PyTorch buffers — they round-trip
  bit-exactly through save/load.

References (APA-7)
------------------
Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen,
    D., & Yih, W. (2020). Dense passage retrieval for open-domain
    question answering. In Proceedings of the 2020 Conference on
    Empirical Methods in Natural Language Processing (pp. 6769-6781).

Kraus, M., Feuerriegel, S., & Oztekin, A. (2020). Deep learning in
    business analytics and operations research: Models, applications
    and managerial implications. European Journal of Operational
    Research, 281(3), 628-641.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal,
    N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S.,
    & Kiela, D. (2020). Retrieval-augmented generation for knowledge-
    intensive NLP tasks. In Advances in Neural Information Processing
    Systems 33 (pp. 9459-9474).

Oord, A. van den, Li, Y., & Vinyals, O. (2018). Representation
    learning with contrastive predictive coding. arXiv:1807.03748.

Schölkopf, B., & Smola, A. J. (2002). Learning with kernels: Support
    vector machines, regularization, optimization, and beyond. MIT
    Press. [For Mahalanobis-PSD background.]

Weinberger, K. Q., & Saul, L. K. (2009). Distance metric learning for
    large margin nearest neighbor classification. Journal of Machine
    Learning Research, 10, 207-244.

Version
-------
1.0.0  Camera-ready KBS submission.

       Initial schema. Future revisions will bump SCHEMA_VERSION and
       must implement an explicit migration path documented in the
       AnalogyEngine.load method.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import warnings
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
    DEFAULT_FEATURE_NAMES,
    DEFAULT_LEVERS,
    DEFAULT_SIGNATURE_DIM,
    INSTITUTIONS,
    SCHEMA_VERSION as CASE_SCHEMA_VERSION,
    CaseLibrary,
    CaseLibraryError,
    CaseMemoryError,
    CaseSignatureEncoder,
    CrisisCase,
    build_signature_matrix,
)


__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "DEFAULT_K_RETRIEVAL",
    "DEFAULT_METRIC_RANK",
    "DEFAULT_RETRIEVAL_TEMPERATURE",
    "DEFAULT_CONTEXT_DIM",
    "POLICY_FINGERPRINT_DIM",
    "OUTCOME_FINGERPRINT_DIM",
    # Exceptions
    "AnalogyEngineError",
    "RetrieverNotFittedError",
    "LibraryChecksumMismatchError",
    # Result
    "RetrievalResult",
    # Components
    "ConditionalRetriever",
    "CaseContextEncoder",
    "AnalogyEngine",
    # Helpers
    "coordinator_posterior_to_crisis_posterior",
    "compute_policy_fingerprint",
    "compute_outcome_fingerprint",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Semantic version of the on-disk analogy-engine artefact.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: Default number of historical analogues to retrieve per query.
DEFAULT_K_RETRIEVAL: Final[int] = 5

#: Default rank of the per-type Cholesky factor L_τ in the conditional
#: retrieval metric. With signature_dim=32 and rank=16, each W_τ has
#: 32 × 16 = 512 parameters; across the 5 crisis types, 2,560 retriever
#: parameters total. Half of full rank is a defensible default —
#: enough capacity to differentiate crisis types, conservative enough
#: to avoid overfitting the small EM crisis library.
DEFAULT_METRIC_RANK: Final[int] = 16

#: Default softmax temperature when computing top-K aggregation weights
#: from raw similarity scores. Lower values sharpen the top-K weights
#: (concentrate context on the highest-scoring analogue); higher values
#: spread context across the K. 0.5 is a sensible default.
DEFAULT_RETRIEVAL_TEMPERATURE: Final[float] = 0.5

#: Default dimensionality of the context vector produced by
#: ``CaseContextEncoder.forward``. 64 matches a typical actor input
#: head; the actor projects it to the hidden_dim of the actor MLP.
DEFAULT_CONTEXT_DIM: Final[int] = 64

#: Policy fingerprint dimensionality: one cell per (institution, lever)
#: pair. The canonical EM panel has 3 institutions × 8 levers = 24
#: cells. Each cell holds the signed total magnitude of policy actions
#: in that cell over the case's timeline, in the units native to that
#: lever (bp_change, pp_of_gdp, etc.); cross-cell comparability is
#: established at the standardisation step inside
#: ``CaseContextEncoder.fit``.
POLICY_FINGERPRINT_DIM: Final[int] = len(INSTITUTIONS) * len(DEFAULT_LEVERS)

#: Outcome fingerprint dimensionality: ``output_loss_cumulative_gdp``
#: (the canonical Laeven-Valencia output-loss measure) plus a max-
#: drawdown proxy (the most-negative quarterly ``output_gap`` value
#: observed in the case's post-onset trajectory). Two scalars are
#: enough to distinguish "deep-but-short" from "shallow-but-long"
#: crises in the actor's context — richer outcome features can be
#: added in v1.1 without breaking the v1.0 schema.
OUTCOME_FINGERPRINT_DIM: Final[int] = 2

#: Tolerance for simplex-validation of the type posterior.
_SIMPLEX_ATOL: Final[float] = 1e-4

#: Minimum library size for retriever fit. Below this we cannot
#: assemble both positive and negative pairs reliably.
_MIN_LIBRARY_SIZE_FOR_FIT: Final[int] = 4

#: Filename of the engine manifest inside the engine directory.
_ENGINE_MANIFEST_NAME: Final[str] = "engine_manifest.json"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class AnalogyEngineError(CaseMemoryError):
    """Base class for analogy-engine exceptions. Inherits from
    ``CaseMemoryError`` so callers can catch both with a single
    ``except CaseMemoryError`` clause."""


class RetrieverNotFittedError(AnalogyEngineError):
    """An inference operation was called before ``fit()``."""


class LibraryChecksumMismatchError(AnalogyEngineError):
    """The library passed at load time has a different content
    checksum than the library against which the engine was fit. In
    strict mode this is an error; in lenient mode the engine warns
    and proceeds. Always indicates one of: (a) the user updated the
    library after engine training, (b) the user paired the wrong
    library with the wrong engine, or (c) a case file was tampered
    with."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Stable, sorted, compact JSON for hashing. Duplicates the helper
    in ``case_memory.py`` to keep this module self-contained."""
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
    """Same compatibility policy as ``case_memory._check_compatible_version``:
    MAJOR mismatch raises, MINOR mismatch warns, PATCH mismatch silent."""
    try:
        found = tuple(int(p) for p in found_version.split("."))
        expected = tuple(int(p) for p in expected_version.split("."))
    except (ValueError, AttributeError) as exc:
        raise AnalogyEngineError(
            f"{context}: malformed schema_version {found_version!r}; "
            f"expected MAJOR.MINOR.PATCH"
        ) from exc
    if len(found) != 3 or len(expected) != 3:
        raise AnalogyEngineError(
            f"{context}: schema_version must be MAJOR.MINOR.PATCH; "
            f"got {found_version!r}"
        )
    if found[0] != expected[0]:
        raise AnalogyEngineError(
            f"{context}: schema major-version mismatch (found "
            f"{found_version!r}, code supports {expected_version!r})."
        )
    if found[1] != expected[1]:
        logger.warning(
            "%s: schema minor-version mismatch (found %s, code supports %s)",
            context, found_version, expected_version,
        )


def _check_simplex(p: np.ndarray, atol: float = _SIMPLEX_ATOL,
                   context: str = "posterior") -> np.ndarray:
    """Validate that ``p`` is a non-negative vector summing to 1 within
    ``atol``. Negative entries (beyond ``-atol``) raise; non-unit sum
    warns and renormalises. Returns the normalised vector."""
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.ndim != 1:
        raise ValueError(
            f"{context}: must be a 1-D vector; got shape {p_arr.shape}"
        )
    if np.any(~np.isfinite(p_arr)):
        raise ValueError(f"{context}: contains non-finite entries")
    if np.any(p_arr < -atol):
        raise ValueError(
            f"{context}: has negative entries (min={float(p_arr.min())})"
        )
    p_clipped = np.clip(p_arr, 0.0, None)
    s = float(p_clipped.sum())
    if s <= 0:
        raise ValueError(
            f"{context}: sums to {s}; cannot normalise to simplex"
        )
    if abs(s - 1.0) > atol:
        warnings.warn(
            f"{context} sums to {s:.6f} (not 1); renormalising",
            stacklevel=3,
        )
    return p_clipped / s


def compute_policy_fingerprint(case: CrisisCase) -> np.ndarray:
    """Construct a ``(POLICY_FINGERPRINT_DIM,)`` vector summarising a
    case's policy timeline.

    The fingerprint is the flattened ``(n_institutions, n_levers)``
    matrix in which cell ``(i, j)`` holds the signed sum of action
    magnitudes whose institution is ``INSTITUTIONS[i]`` and whose
    lever is ``DEFAULT_LEVERS[j]``. The flattening uses row-major
    (institution-major) order. Cells with no actions are zero. Units
    within a cell are typically consistent (e.g., all
    ``(central_bank, policy_rate)`` actions use ``bp_change``); cross-
    cell comparability is established at the standardisation step
    inside ``CaseContextEncoder.fit``.

    Notes
    -----
    This function is intentionally deterministic and does not depend
    on any learned state. It can be inspected and verified against
    the source ``.case.json`` file with a calculator. Reviewers and
    central-bank users can therefore audit the policy summary that
    the model conditions on without re-running the encoder.
    """
    fp = np.zeros((len(INSTITUTIONS), len(DEFAULT_LEVERS)), dtype=np.float64)
    inst_idx = {inst: i for i, inst in enumerate(INSTITUTIONS)}
    lever_idx = {lever: i for i, lever in enumerate(DEFAULT_LEVERS)}
    for action in case.policy_timeline:
        # ``PolicyAction.__post_init__`` guarantees institution and
        # lever are members of the canonical vocabularies; we do not
        # re-validate here.
        i = inst_idx[action.institution]
        j = lever_idx[action.lever]
        fp[i, j] += action.value
    return fp.flatten()


def compute_outcome_fingerprint(case: CrisisCase) -> np.ndarray:
    """Construct a ``(OUTCOME_FINGERPRINT_DIM,)`` vector summarising a
    case's realised outcome.

    Coordinate 0: ``output_loss_cumulative_gdp`` (in percent of GDP;
    the canonical Laeven-Valencia output-loss measure).
    Coordinate 1: most-negative single-quarter ``output_gap`` observed
    in the post-onset trajectory (a crude max-drawdown proxy).

    For ongoing cases whose post-onset trajectory does not yet cover
    the full ``DEFAULT_POST_ONSET_QUARTERS`` (e.g., Lebanon 2019-),
    the drawdown is computed over whatever data exists. The
    cumulative-loss field is taken as supplied by the case author.
    """
    output_gap_idx = DEFAULT_FEATURE_NAMES.index("output_gap")
    post = case.post_onset_trajectory
    if post.shape[0] > 0:
        max_drawdown = float(post[:, output_gap_idx].min())
    else:
        max_drawdown = 0.0
    return np.array(
        [float(case.output_loss_cumulative_gdp), max_drawdown],
        dtype=np.float64,
    )


def coordinator_posterior_to_crisis_posterior(
    coord_posterior: np.ndarray,
) -> np.ndarray:
    """Convert a (n_coord_types,)-dim coordinator posterior to a
    (len(CRISIS_TYPES),)-dim crisis-type posterior over the case-library
    vocabulary.

    The coordinator's output (from ``coordinator.py``) is a posterior
    over ``("none", "banking", "currency", "sovereign", "twin",
    "triple")`` — six entries including ``"none"``. The case library
    has no ``"none"`` cases by construction, so the analogy engine's
    retriever conditions only on the five crisis-type entries.

    This helper drops the ``"none"`` entry (index 0) and renormalises
    the remaining five. If the result has zero mass (the coordinator
    is entirely certain there is no crisis), the helper returns a
    uniform posterior over crisis types: the retriever should still
    function in degenerate cases, though the caller may prefer to
    gate retrieval on a separate crisis-probability threshold.
    """
    p = np.asarray(coord_posterior, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(
            f"coordinator posterior must be 1-D; got shape {p.shape}"
        )
    if p.shape[0] != len(CRISIS_TYPES) + 1:
        raise ValueError(
            f"coordinator posterior must have {len(CRISIS_TYPES) + 1} "
            f"entries (none + {len(CRISIS_TYPES)} crisis types); "
            f"got {p.shape[0]}"
        )
    p_crisis = p[1:]
    s = float(p_crisis.sum())
    if s <= 1e-12:
        logger.warning(
            "coordinator posterior places ~all mass on 'none'; returning "
            "uniform crisis-type posterior. Caller should gate retrieval "
            "on a separate crisis-probability threshold."
        )
        return np.ones_like(p_crisis) / len(p_crisis)
    return p_crisis / s


# ---------------------------------------------------------------------------
# RetrievalResult: one retrieval call's full audit trail
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalResult:
    """Immutable record of one retrieval call.

    A central-bank desk officer reading a policy recommendation sees
    this object's contents: which historical cases were retrieved, how
    similar they were, and how much weight each one received in the
    aggregated context that informed the action. The object can be
    persisted as JSON alongside the action for the audit trail.

    Fields
    ------
    case_ids : tuple of str
        Top-K case_ids in retrieved order (highest similarity first).
    similarities : np.ndarray, shape (K,)
        Raw retriever scores for the K retrieved cases, in the same
        order as ``case_ids``.
    weights : np.ndarray, shape (K,)
        Softmax-normalised weights over the K retrieved cases (sum to 1).
        These are the weights used to aggregate per-case context features
        into the final ``context_vector``.
    type_posterior : np.ndarray, shape (len(CRISIS_TYPES),)
        The simplex posterior over crisis types that conditioned this
        retrieval. Stored so the audit trail records the model's
        belief at the moment of retrieval.
    query_signature : np.ndarray, shape (signature_dim,)
        The L2-normalised query signature produced by the
        ``CaseSignatureEncoder`` from the live macro trajectory.
    context_vector : np.ndarray, shape (context_dim,)
        The aggregated context vector consumed by the downstream
        MARL actor.
    temperature : float
        The softmax temperature used to convert similarities to
        weights. Recorded for reproducibility.
    """

    case_ids: tuple[str, ...]
    similarities: np.ndarray
    weights: np.ndarray
    type_posterior: np.ndarray
    query_signature: np.ndarray
    context_vector: np.ndarray
    temperature: float = DEFAULT_RETRIEVAL_TEMPERATURE

    def __post_init__(self) -> None:
        k = len(self.case_ids)
        if k == 0:
            raise ValueError("RetrievalResult: case_ids must be non-empty")
        for name, arr, expected_shape in (
            ("similarities", self.similarities, (k,)),
            ("weights", self.weights, (k,)),
            ("type_posterior", self.type_posterior, (len(CRISIS_TYPES),)),
        ):
            a = np.asarray(arr)
            if a.shape != expected_shape:
                raise ValueError(
                    f"RetrievalResult.{name}: expected shape "
                    f"{expected_shape}; got {a.shape}"
                )
        if not np.all(np.isfinite(self.similarities)):
            raise ValueError("RetrievalResult.similarities has non-finite entries")
        if np.any(self.weights < -1e-6):
            raise ValueError("RetrievalResult.weights has negative entries")
        if abs(float(self.weights.sum()) - 1.0) > 1e-3:
            raise ValueError(
                f"RetrievalResult.weights sum to "
                f"{float(self.weights.sum())}; should be 1"
            )
        if self.query_signature.ndim != 1 or self.query_signature.size == 0:
            raise ValueError("RetrievalResult.query_signature must be 1-D non-empty")
        if self.context_vector.ndim != 1 or self.context_vector.size == 0:
            raise ValueError("RetrievalResult.context_vector must be 1-D non-empty")
        # Defensive: mark arrays read-only
        for arr in (self.similarities, self.weights, self.type_posterior,
                    self.query_signature, self.context_vector):
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)

    @property
    def k(self) -> int:
        """Number of retrieved cases."""
        return len(self.case_ids)

    @property
    def top1_case_id(self) -> str:
        """The single most-similar retrieved case."""
        return self.case_ids[0]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation. Numpy arrays become Python
        lists of floats."""
        return {
            "case_ids": list(self.case_ids),
            "similarities": [float(v) for v in self.similarities],
            "weights": [float(v) for v in self.weights],
            "type_posterior": [float(v) for v in self.type_posterior],
            "query_signature": [float(v) for v in self.query_signature],
            "context_vector": [float(v) for v in self.context_vector],
            "temperature": float(self.temperature),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        if indent is None:
            return _canonical_json(self.to_dict())
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=indent,
            ensure_ascii=True, allow_nan=False,
        )

    def __repr__(self) -> str:
        top = self.case_ids[:3]
        more = f", ...+{self.k - 3}" if self.k > 3 else ""
        return (f"RetrievalResult(top1={self.top1_case_id!r}, "
                f"k={self.k}, retrieved=[{', '.join(repr(c) for c in top)}{more}])")


# ---------------------------------------------------------------------------
# ConditionalRetriever: crisis-type-conditional similarity metric
# ---------------------------------------------------------------------------


class ConditionalRetriever(nn.Module):
    """Crisis-type-conditional retrieval metric.

    The metric is a soft mixture over crisis types of per-type
    Mahalanobis-style bilinear forms:

    .. math::

        s(\\mathbf{x}, \\mathbf{c} \\mid \\boldsymbol{\\pi})
            = \\sum_{\\tau} \\pi_\\tau \\cdot \\mathbf{x}^{\\top}
              W_\\tau \\mathbf{c}

    where :math:`\\boldsymbol{\\pi}` is a simplex over crisis types and
    each :math:`W_\\tau` is a symmetric positive-(semi-)definite matrix
    parameterised via a low-rank Cholesky factor:

    .. math::

        W_\\tau = L_\\tau L_\\tau^{\\top} + \\gamma I

    where :math:`L_\\tau \\in \\mathbb{R}^{D \\times r}` is the per-type
    learnable factor, :math:`\\gamma > 0` is a shared positive scalar
    on the identity baseline (kept positive by parameterising it as
    :math:`\\gamma = \\exp(\\log\\gamma)`), and :math:`r` is the
    factorisation rank (``rank`` parameter; default 16).

    At initialisation we set :math:`L_\\tau \\approx 0` and
    :math:`\\gamma = 1`, so the metric starts as cosine similarity
    (since the signatures are L2-normalised). Training pushes the
    :math:`L_\\tau` away from zero in directions that distinguish
    crisis types — providing the inductive bias that "if we cannot
    learn anything useful, fall back to cosine similarity."

    Efficient evaluation
    --------------------
    Because the metric is linear in :math:`\\boldsymbol{\\pi}`,
    soft mixing is mathematically equivalent to:

    .. math::

        s(\\mathbf{x}, \\mathbf{c} \\mid \\boldsymbol{\\pi})
            = \\sum_\\tau \\pi_\\tau
              \\langle L_\\tau^{\\top} \\mathbf{x},
                       L_\\tau^{\\top} \\mathbf{c} \\rangle
              + \\gamma \\langle \\mathbf{x}, \\mathbf{c} \\rangle

    (the :math:`\\gamma`-term pulls out of the sum because
    :math:`\\sum_\\tau \\pi_\\tau = 1`). Implemented as a single
    ``torch.einsum`` over ``(B, T, R)`` projections plus a
    cosine-similarity matrix multiply.

    Parameter count
    ---------------
    ``T * D * r + 1`` parameters total. For the canonical configuration
    (T=5, D=32, r=16) this is 2,561 parameters: small enough that
    overfitting on a ~150-case library is mitigated by appropriate
    weight decay, large enough to encode meaningfully different
    metrics across the five crisis types.

    Notes
    -----
    The retriever is intentionally agnostic to the *query encoder*:
    it consumes pre-computed signature vectors (L2-normalised) as
    its input and treats them as fixed during its own training. The
    upstream ``CaseSignatureEncoder`` is trained in Phase B of the
    orchestrator; the retriever is trained in Phase C with the encoder
    frozen. Phase E (optional joint fine-tune) can co-train them.
    """

    def __init__(
        self,
        *,
        signature_dim: int = DEFAULT_SIGNATURE_DIM,
        n_types: int = len(CRISIS_TYPES),
        rank: int = DEFAULT_METRIC_RANK,
        init_L_scale: float = 0.01,
        init_gamma: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if signature_dim < 2:
            raise ValueError(f"signature_dim must be >= 2; got {signature_dim}")
        if n_types < 1:
            raise ValueError(f"n_types must be >= 1; got {n_types}")
        if rank < 1 or rank > signature_dim:
            raise ValueError(
                f"rank must be in [1, signature_dim={signature_dim}]; "
                f"got {rank}"
            )
        if init_L_scale <= 0:
            raise ValueError(f"init_L_scale must be > 0; got {init_L_scale}")
        if init_gamma <= 0:
            raise ValueError(f"init_gamma must be > 0; got {init_gamma}")

        self.config: dict[str, Any] = {
            "signature_dim": int(signature_dim),
            "n_types": int(n_types),
            "rank": int(rank),
            "init_L_scale": float(init_L_scale),
            "init_gamma": float(init_gamma),
            "seed": int(seed),
        }

        # Deterministic init via global torch RNG; save+restore so we
        # don't pollute the caller's RNG state. Same pattern as
        # CaseSignatureEncoder.
        _saved_cpu = torch.get_rng_state()
        _saved_cuda = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            # L: (n_types, signature_dim, rank); small Gaussian
            # initialisation so W_τ ≈ γ I at start.
            self.L = nn.Parameter(
                torch.randn(n_types, signature_dim, rank) * init_L_scale
            )
            # log_gamma: scalar; parameterised in log-space so γ stays
            # positive without a constraint. exp(log γ) = γ.
            self.log_gamma = nn.Parameter(
                torch.tensor(math.log(init_gamma), dtype=torch.float32)
            )
        finally:
            torch.set_rng_state(_saved_cpu)
            if _saved_cuda is not None:
                torch.cuda.set_rng_state_all(_saved_cuda)

        self._is_fitted: bool = False
        self._fit_history: dict[str, list[float]] = {}

    @property
    def gamma(self) -> torch.Tensor:
        """The current value of γ as a scalar tensor."""
        return torch.exp(self.log_gamma)

    # ------------------------------------------------------------------ #
    # Forward pass: similarity score matrix
    # ------------------------------------------------------------------ #

    def forward(
        self,
        query_sigs: torch.Tensor,
        case_sigs: torch.Tensor,
        type_posterior: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the (B, N) similarity matrix.

        Parameters
        ----------
        query_sigs : Tensor of shape (B, D)
            Query signatures (L2-normalised; caller's responsibility).
        case_sigs : Tensor of shape (N, D)
            Library case signatures (L2-normalised).
        type_posterior : Tensor of shape (B, T)
            Per-query simplex over crisis types.

        Returns
        -------
        scores : Tensor of shape (B, N)
        """
        if query_sigs.ndim != 2:
            raise ValueError(
                f"query_sigs must be 2-D (B, D); got shape "
                f"{tuple(query_sigs.shape)}"
            )
        if case_sigs.ndim != 2:
            raise ValueError(
                f"case_sigs must be 2-D (N, D); got shape "
                f"{tuple(case_sigs.shape)}"
            )
        if type_posterior.ndim != 2:
            raise ValueError(
                f"type_posterior must be 2-D (B, T); got shape "
                f"{tuple(type_posterior.shape)}"
            )
        B, D = query_sigs.shape
        N, D_c = case_sigs.shape
        B_p, T = type_posterior.shape
        if D != D_c:
            raise ValueError(
                f"signature dim mismatch: query {D} vs case {D_c}"
            )
        if D != self.config["signature_dim"]:
            raise ValueError(
                f"input signature_dim {D} != retriever config "
                f"{self.config['signature_dim']}"
            )
        if B != B_p:
            raise ValueError(
                f"batch mismatch: query has B={B}, posterior has B={B_p}"
            )
        if T != self.config["n_types"]:
            raise ValueError(
                f"posterior has T={T}; retriever expects "
                f"{self.config['n_types']}"
            )

        # Project query and case sigs into the per-type rank-r spaces.
        # L has shape (T, D, R); einsum gives (B, T, R) and (N, T, R).
        q_proj = torch.einsum("bd,tdr->btr", query_sigs, self.L)   # (B, T, R)
        c_proj = torch.einsum("nd,tdr->ntr", case_sigs, self.L)    # (N, T, R)

        # Per-type bilinear scores: (B, T, N) via einsum.
        per_type_sim = torch.einsum("btr,ntr->btn", q_proj, c_proj)

        # Soft mixing over types: weight by posterior, sum over T.
        weighted = torch.einsum("bt,btn->bn", type_posterior, per_type_sim)

        # Add gamma * cosine baseline. Because the signatures are
        # L2-normalised by convention, q @ c.T already IS the cosine-
        # similarity matrix; we trust the caller and skip re-normalising
        # for efficiency. (The encoder always emits unit-norm vectors.)
        cosine = query_sigs @ case_sigs.T   # (B, N)
        return weighted + self.gamma * cosine

    # ------------------------------------------------------------------ #
    # Top-K retrieval with both soft (training) and hard (inference) paths
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query_sigs: torch.Tensor,
        case_sigs: torch.Tensor,
        type_posterior: torch.Tensor,
        *,
        k: int = DEFAULT_K_RETRIEVAL,
        temperature: float = DEFAULT_RETRIEVAL_TEMPERATURE,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hard top-K retrieval with softmax-normalised aggregation weights.

        Parameters
        ----------
        query_sigs : Tensor (B, D)
        case_sigs : Tensor (N, D)
        type_posterior : Tensor (B, T)
        k : int, default DEFAULT_K_RETRIEVAL
            Top-K to return. Automatically capped at N if k > N.
        temperature : float, default DEFAULT_RETRIEVAL_TEMPERATURE
            Softmax temperature for the aggregation weights.

        Returns
        -------
        top_idx : LongTensor (B, k)
            Indices into ``case_sigs`` of the top-K retrievals.
        top_sims : Tensor (B, k)
            Raw similarity scores for the top-K, in the same order.
        weights : Tensor (B, k)
            Softmax weights over the top-K (rows sum to 1), used for
            context aggregation.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        scores = self.forward(query_sigs, case_sigs, type_posterior)
        n_cases = case_sigs.shape[0]
        if k > n_cases:
            logger.warning(
                "retrieve: k=%d > n_cases=%d; capping k=n_cases", k, n_cases,
            )
            k = n_cases
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        top_sims, top_idx = scores.topk(k, dim=-1)
        weights = F.softmax(top_sims / temperature, dim=-1)
        return top_idx, top_sims, weights

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> Path:
        """Save state_dict + config to ``path`` (a directory). Atomic
        per file."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sd_path = path / "retriever.pt"
        sd_tmp = sd_path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), sd_tmp)
        os.replace(sd_tmp, sd_path)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "config": self.config,
            "is_fitted": self._is_fitted,
            "fit_history_summary": (
                {
                    "n_epochs": len(self._fit_history.get("loss", [])),
                    "final_loss": (
                        self._fit_history["loss"][-1]
                        if self._fit_history.get("loss") else None
                    ),
                }
                if self._fit_history else {}
            ),
        }
        _atomic_write_text(
            path / "retriever_config.json",
            json.dumps(meta, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "ConditionalRetriever":
        path = Path(path)
        meta_path = path / "retriever_config.json"
        sd_path = path / "retriever.pt"
        if not meta_path.is_file() or not sd_path.is_file():
            raise AnalogyEngineError(
                f"ConditionalRetriever.load: missing retriever.pt or "
                f"retriever_config.json under {path}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _check_compatible_version(
            meta.get("schema_version", SCHEMA_VERSION),
            context=f"ConditionalRetriever@{path}",
        )
        retr = cls(**meta["config"])
        retr.load_state_dict(torch.load(sd_path, map_location=map_location))
        retr._is_fitted = bool(meta.get("is_fitted", False))
        retr.to(map_location)
        retr.eval()
        return retr


# ---------------------------------------------------------------------------
# CaseContextEncoder: turn retrieved cases into a fixed-size context vector
# ---------------------------------------------------------------------------


class CaseContextEncoder(nn.Module):
    """Encodes retrieved cases into a fixed-size context vector for the
    downstream MARL actor.

    Each retrieved case contributes three feature blocks:

    * Signature (``signature_dim`` dims) — already L2-normalised by
      the upstream ``CaseSignatureEncoder``.
    * Policy fingerprint (``POLICY_FINGERPRINT_DIM`` dims, =24 for the
      canonical configuration) — signed sum of action magnitudes per
      ``(institution, lever)`` cell, then standardised per-cell using
      the training library's mean and std.
    * Outcome fingerprint (``OUTCOME_FINGERPRINT_DIM`` dims, =2) —
      cumulative output-loss and post-onset max-drawdown, standardised.

    The per-case feature vectors are aggregated across the K retrieved
    cases using the retrieval-weighted average; the result is passed
    through a two-layer MLP that projects to ``context_dim``. The
    aggregation is linear (weighted sum) so the actor sees a smooth
    function of retrieval weights — useful for end-to-end fine-tuning.

    The standardisation statistics are computed at ``fit`` time over
    the entire training library and stored as PyTorch buffers
    (``policy_mean``, ``policy_std``, ``outcome_mean``, ``outcome_std``).
    They are part of the state_dict and round-trip bit-exactly through
    save/load.
    """

    def __init__(
        self,
        *,
        signature_dim: int = DEFAULT_SIGNATURE_DIM,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if signature_dim < 2:
            raise ValueError(f"signature_dim must be >= 2; got {signature_dim}")
        if context_dim < 2:
            raise ValueError(f"context_dim must be >= 2; got {context_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")

        self.config: dict[str, Any] = {
            "signature_dim": int(signature_dim),
            "context_dim": int(context_dim),
            "dropout": float(dropout),
            "seed": int(seed),
        }
        self.case_feature_dim: int = (
            signature_dim + POLICY_FINGERPRINT_DIM + OUTCOME_FINGERPRINT_DIM
        )

        # Standardisation buffers; populated by .fit (or by load).
        # Initialised so an unfitted encoder produces the raw inputs
        # (mean 0, std 1), with a one-time warning emitted on first
        # forward if unfitted.
        self.register_buffer(
            "policy_mean", torch.zeros(POLICY_FINGERPRINT_DIM)
        )
        self.register_buffer(
            "policy_std", torch.ones(POLICY_FINGERPRINT_DIM)
        )
        self.register_buffer(
            "outcome_mean", torch.zeros(OUTCOME_FINGERPRINT_DIM)
        )
        self.register_buffer(
            "outcome_std", torch.ones(OUTCOME_FINGERPRINT_DIM)
        )

        # Projection MLP. Seeded init via save/restore of global RNG.
        _saved_cpu = torch.get_rng_state()
        _saved_cuda = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            self.proj = nn.Sequential(
                nn.Linear(self.case_feature_dim, context_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_dim, context_dim),
            )
        finally:
            torch.set_rng_state(_saved_cpu)
            if _saved_cuda is not None:
                torch.cuda.set_rng_state_all(_saved_cuda)

        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Fingerprint standardisation
    # ------------------------------------------------------------------ #

    def fit(
        self,
        library: CaseLibrary,
        *,
        eps: float = 1e-6,
    ) -> dict[str, Any]:
        """Compute and store standardisation stats from ``library``.

        Returns a small summary dict suitable for logging."""
        if len(library) < 2:
            raise CaseLibraryError(
                f"CaseContextEncoder.fit: need >= 2 cases to compute "
                f"standardisation; got {len(library)}"
            )

        case_ids = library.case_ids()
        pfps = np.stack([
            compute_policy_fingerprint(library[cid]) for cid in case_ids
        ], axis=0)
        ofps = np.stack([
            compute_outcome_fingerprint(library[cid]) for cid in case_ids
        ], axis=0)

        pol_mean = pfps.mean(axis=0)
        pol_std = pfps.std(axis=0).clip(min=eps)
        out_mean = ofps.mean(axis=0)
        out_std = ofps.std(axis=0).clip(min=eps)

        self.policy_mean.copy_(torch.tensor(pol_mean, dtype=self.policy_mean.dtype))
        self.policy_std.copy_(torch.tensor(pol_std, dtype=self.policy_std.dtype))
        self.outcome_mean.copy_(torch.tensor(out_mean, dtype=self.outcome_mean.dtype))
        self.outcome_std.copy_(torch.tensor(out_std, dtype=self.outcome_std.dtype))
        self._is_fitted = True

        summary = {
            "n_cases": len(library),
            "policy_mean_norm": float(np.linalg.norm(pol_mean)),
            "policy_std_min": float(pol_std.min()),
            "policy_std_max": float(pol_std.max()),
            "outcome_mean": [float(v) for v in out_mean],
            "outcome_std": [float(v) for v in out_std],
        }
        logger.info(
            "CaseContextEncoder.fit: standardisation stats computed from "
            "%d cases; policy_std in [%.3g, %.3g]",
            len(library), summary["policy_std_min"], summary["policy_std_max"],
        )
        return summary

    def standardise_policy(self, raw: torch.Tensor) -> torch.Tensor:
        """Z-score a raw policy-fingerprint tensor of shape
        ``(..., POLICY_FINGERPRINT_DIM)``."""
        return (raw - self.policy_mean) / self.policy_std

    def standardise_outcome(self, raw: torch.Tensor) -> torch.Tensor:
        """Z-score a raw outcome-fingerprint tensor of shape
        ``(..., OUTCOME_FINGERPRINT_DIM)``."""
        return (raw - self.outcome_mean) / self.outcome_std

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        case_features: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate per-case features into a context vector.

        Parameters
        ----------
        case_features : Tensor of shape (B, K, case_feature_dim)
            Stacked per-case feature vectors. The last dim must equal
            ``self.case_feature_dim`` = signature_dim + 24 + 2.
            **Already standardised** — the caller (typically the
            ``AnalogyEngine``) is expected to have applied
            ``standardise_policy`` and ``standardise_outcome`` to the
            policy and outcome blocks. The standardisation buffers
            remain on the module so the caller can invoke them; we do
            not re-standardise inside ``forward`` to avoid silent
            double application.
        weights : Tensor of shape (B, K)
            Softmax weights over the K retrieved cases (rows sum to 1).
            Negative values raise; non-unit row sums are tolerated
            within ``_SIMPLEX_ATOL`` but warned beyond it.

        Returns
        -------
        context : Tensor of shape (B, context_dim)
        """
        if not self._is_fitted:
            warnings.warn(
                "CaseContextEncoder.forward: standardisation buffers are at "
                "default values (0/1). Call .fit on the training library "
                "first or the actor will see un-standardised fingerprints.",
                stacklevel=2,
            )

        if case_features.ndim != 3:
            raise ValueError(
                f"case_features must be 3-D (B, K, D); got shape "
                f"{tuple(case_features.shape)}"
            )
        if case_features.shape[-1] != self.case_feature_dim:
            raise ValueError(
                f"case_features last dim {case_features.shape[-1]} != "
                f"expected {self.case_feature_dim}"
            )
        if weights.ndim != 2 or weights.shape != case_features.shape[:2]:
            raise ValueError(
                f"weights shape {tuple(weights.shape)} does not match "
                f"case_features.shape[:2]={tuple(case_features.shape[:2])}"
            )
        if torch.any(weights < -1e-6):
            raise ValueError("weights has negative entries")
        # Row-sum check (cheap; warn if off)
        row_sums = weights.sum(dim=-1)
        if torch.any((row_sums - 1.0).abs() > 1e-3):
            warnings.warn(
                f"weights row sums deviate from 1 "
                f"(max deviation {(row_sums - 1).abs().max().item():.4f}); "
                f"forward proceeds but caller should renormalise",
                stacklevel=2,
            )

        # Weighted sum over K → (B, case_feature_dim)
        pooled = (case_features * weights.unsqueeze(-1)).sum(dim=1)
        return self.proj(pooled)

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sd_path = path / "context_encoder.pt"
        sd_tmp = sd_path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), sd_tmp)
        os.replace(sd_tmp, sd_path)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "config": self.config,
            "is_fitted": self._is_fitted,
        }
        _atomic_write_text(
            path / "context_encoder_config.json",
            json.dumps(meta, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "CaseContextEncoder":
        path = Path(path)
        meta_path = path / "context_encoder_config.json"
        sd_path = path / "context_encoder.pt"
        if not meta_path.is_file() or not sd_path.is_file():
            raise AnalogyEngineError(
                f"CaseContextEncoder.load: missing files under {path}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _check_compatible_version(
            meta.get("schema_version", SCHEMA_VERSION),
            context=f"CaseContextEncoder@{path}",
        )
        enc = cls(**meta["config"])
        enc.load_state_dict(torch.load(sd_path, map_location=map_location))
        enc._is_fitted = bool(meta.get("is_fitted", False))
        enc.to(map_location)
        enc.eval()
        return enc


# ---------------------------------------------------------------------------
# AnalogyEngine: top-level orchestrator
# ---------------------------------------------------------------------------


class AnalogyEngine:
    """Top-level orchestrator: case library + signature encoder +
    conditional retriever + case-context encoder.

    Construction precomputes the retrieval index — the signature
    matrix, the standardised policy-fingerprint matrix, and the
    standardised outcome-fingerprint matrix — caching them as
    ``self._signature_matrix`` etc. for fast inference. The library
    itself is held by reference, not copied.

    Lifecycle
    ---------
    >>> # Phase B output: trained signature encoder
    >>> encoder = CaseSignatureEncoder(seed=42)
    >>> encoder.fit(library)
    >>>
    >>> # Phase C: build and train engine
    >>> retriever = ConditionalRetriever()
    >>> ctx_encoder = CaseContextEncoder()
    >>> engine = AnalogyEngine(
    ...     library=library,
    ...     signature_encoder=encoder,
    ...     retriever=retriever,
    ...     context_encoder=ctx_encoder,
    ... )
    >>> engine.fit(n_epochs=200)
    >>>
    >>> # Inference
    >>> coord_posterior = np.array([0.2, 0.1, 0.5, 0.1, 0.1, 0.0])  # 6-dim, with 'none'
    >>> crisis_posterior = coordinator_posterior_to_crisis_posterior(coord_posterior)
    >>> query_traj = current_macro_window  # (20, 12)
    >>> result = engine.retrieve(query_traj, crisis_posterior, k=5)
    >>> print(result.case_ids)        # top-K retrieved
    >>> print(result.weights)         # softmax weights
    >>> actor_context = result.context_vector  # passed to MARL actor

    Library binding
    ---------------
    The engine records the library's content checksum at construction.
    On save, this checksum is written to the engine manifest. On load,
    the caller supplies a library; if the supplied library's checksum
    differs from the saved one, the engine raises (strict mode,
    default) or warns (lenient mode). This catches "wrong library
    paired with wrong engine" failures cleanly. To accept a new
    library, retrain the engine on it.
    """

    def __init__(
        self,
        library: CaseLibrary,
        signature_encoder: CaseSignatureEncoder,
        retriever: ConditionalRetriever,
        context_encoder: CaseContextEncoder,
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if len(library) < 1:
            raise CaseLibraryError(
                "AnalogyEngine: cannot construct over an empty library"
            )
        if signature_encoder.config["signature_dim"] != retriever.config["signature_dim"]:
            raise AnalogyEngineError(
                f"AnalogyEngine: signature_dim mismatch "
                f"(encoder={signature_encoder.config['signature_dim']}, "
                f"retriever={retriever.config['signature_dim']})"
            )
        if signature_encoder.config["signature_dim"] != context_encoder.config["signature_dim"]:
            raise AnalogyEngineError(
                f"AnalogyEngine: signature_dim mismatch "
                f"(encoder={signature_encoder.config['signature_dim']}, "
                f"context_encoder={context_encoder.config['signature_dim']})"
            )
        if retriever.config["n_types"] != len(CRISIS_TYPES):
            raise AnalogyEngineError(
                f"AnalogyEngine: retriever n_types "
                f"{retriever.config['n_types']} != len(CRISIS_TYPES) "
                f"{len(CRISIS_TYPES)}"
            )

        self.library = library
        self.signature_encoder = signature_encoder
        self.retriever = retriever
        self.context_encoder = context_encoder
        self._library_checksum_at_init: str = library.library_checksum()
        self._device: torch.device = torch.device(device)

        # Build retrieval index (signatures + fingerprints + crisis types)
        self._case_ids: tuple[str, ...] = ()
        self._signature_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._policy_fp_raw: np.ndarray = np.empty(
            (0, POLICY_FINGERPRINT_DIM), dtype=np.float64,
        )
        self._outcome_fp_raw: np.ndarray = np.empty(
            (0, OUTCOME_FINGERPRINT_DIM), dtype=np.float64,
        )
        self._crisis_type_labels: np.ndarray = np.empty(0, dtype=np.int64)
        self._fit_history: dict[str, list[float]] = {}
        self._is_fitted: bool = False
        self._build_index()

        # Move modules to device
        self.signature_encoder.to(self._device)
        self.retriever.to(self._device)
        self.context_encoder.to(self._device)

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #

    def _build_index(self) -> None:
        """Precompute signature, policy-fp, outcome-fp, and crisis-type
        index matrices from the library."""
        case_ids, sigs = build_signature_matrix(
            self.library, self.signature_encoder,
        )
        self._case_ids = case_ids
        self._signature_matrix = sigs.astype(np.float32, copy=False)
        if len(case_ids) > 0:
            self._policy_fp_raw = np.stack([
                compute_policy_fingerprint(self.library[cid])
                for cid in case_ids
            ], axis=0)
            self._outcome_fp_raw = np.stack([
                compute_outcome_fingerprint(self.library[cid])
                for cid in case_ids
            ], axis=0)
            self._crisis_type_labels = np.array([
                CRISIS_TYPES.index(self.library[cid].crisis_type)
                for cid in case_ids
            ], dtype=np.int64)
        logger.info(
            "AnalogyEngine: built retrieval index for %d cases "
            "(signature_dim=%d, library_checksum=%s...)",
            len(case_ids), self._signature_matrix.shape[1] if case_ids else 0,
            self._library_checksum_at_init[:12],
        )

    @property
    def n_cases(self) -> int:
        return len(self._case_ids)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return self._case_ids

    @property
    def library_checksum_at_init(self) -> str:
        return self._library_checksum_at_init

    # ------------------------------------------------------------------ #
    # Internal: assemble per-case feature tensor for retrieved indices
    # ------------------------------------------------------------------ #

    def _gather_case_features(
        self,
        top_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Gather (B, K, case_feature_dim) feature tensor for the
        retrieved indices, with policy and outcome blocks standardised
        via the context encoder's buffers.

        ``top_idx`` is a LongTensor of shape (B, K).
        """
        device = top_idx.device
        # Convert numpy index matrices to tensors lazily (cached on CPU,
        # moved when needed). For simplicity we re-tensor each call;
        # the matrices are small.
        sig_mat = torch.as_tensor(
            self._signature_matrix, dtype=torch.float32, device=device,
        )
        pol_mat = torch.as_tensor(
            self._policy_fp_raw, dtype=torch.float32, device=device,
        )
        out_mat = torch.as_tensor(
            self._outcome_fp_raw, dtype=torch.float32, device=device,
        )

        # Standardise policy and outcome blocks using context_encoder buffers
        pol_std = self.context_encoder.standardise_policy(pol_mat)
        out_std = self.context_encoder.standardise_outcome(out_mat)

        # Gather (B, K, _) for each block
        sig_gathered = sig_mat[top_idx]    # (B, K, signature_dim)
        pol_gathered = pol_std[top_idx]    # (B, K, POLICY_FINGERPRINT_DIM)
        out_gathered = out_std[top_idx]    # (B, K, OUTCOME_FINGERPRINT_DIM)

        case_features = torch.cat(
            [sig_gathered, pol_gathered, out_gathered], dim=-1,
        )
        return case_features

    # ------------------------------------------------------------------ #
    # Inference: retrieve for one query
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query_trajectory: np.ndarray,
        type_posterior: np.ndarray,
        *,
        k: int = DEFAULT_K_RETRIEVAL,
        temperature: float = DEFAULT_RETRIEVAL_TEMPERATURE,
        use_type_conditioning: bool = True,
    ) -> RetrievalResult:
        """Retrieve top-K analogous cases for a single query.

        Parameters
        ----------
        query_trajectory : np.ndarray of shape ``(T_pre, N_features)``
            The 20-quarter pre-current macro window.
        type_posterior : np.ndarray of shape ``(len(CRISIS_TYPES),)``
            Crisis-type posterior, on the simplex. If you have a
            coordinator-style posterior including ``"none"``, call
            ``coordinator_posterior_to_crisis_posterior`` first.
        k : int, default DEFAULT_K_RETRIEVAL
        temperature : float, default DEFAULT_RETRIEVAL_TEMPERATURE
        use_type_conditioning : bool, default True
            If False, the retriever's W_τ matrices are skipped and only
            the γ-cosine baseline is used. This is the
            ``no_type_conditioning`` ablation referenced in E03 of the
            paper.

        Returns
        -------
        RetrievalResult
        """
        if not self._is_fitted:
            warnings.warn(
                "AnalogyEngine.retrieve: engine has not been .fit. "
                "Retrieval uses unfitted retriever (≈ cosine similarity) "
                "and unfitted context standardisation. Call .fit first.",
                stacklevel=2,
            )

        # Validate inputs
        type_post = _check_simplex(type_posterior, context="type_posterior")
        if type_post.shape[0] != len(CRISIS_TYPES):
            raise ValueError(
                f"type_posterior length {type_post.shape[0]} != "
                f"len(CRISIS_TYPES) = {len(CRISIS_TYPES)}. If you have a "
                f"coordinator output including 'none', call "
                f"coordinator_posterior_to_crisis_posterior first."
            )
        query_traj = np.asarray(query_trajectory, dtype=np.float64)
        if query_traj.ndim != 2:
            raise ValueError(
                f"query_trajectory must be 2-D; got shape {query_traj.shape}"
            )

        # 1) encode the query
        q_sig_np = self.signature_encoder.encode(query_traj)
        # encode() returns (D,) for single-trajectory input
        if q_sig_np.ndim != 1:
            raise AnalogyEngineError(
                f"signature_encoder.encode returned shape {q_sig_np.shape}; "
                f"expected 1-D"
            )

        # 2) score against the library
        q_sig = torch.as_tensor(
            q_sig_np, dtype=torch.float32, device=self._device,
        ).unsqueeze(0)   # (1, D)
        case_sigs = torch.as_tensor(
            self._signature_matrix, dtype=torch.float32, device=self._device,
        )   # (N, D)
        type_post_t = torch.as_tensor(
            type_post, dtype=torch.float32, device=self._device,
        ).unsqueeze(0)   # (1, T)

        if use_type_conditioning:
            # Retrieval is an inference op — modules must be in eval mode
            # so any dropout layers are identity. Save the training state
            # and restore on exit so we don't disturb a fit-in-progress.
            was_training_retriever = self.retriever.training
            was_training_context = self.context_encoder.training
            self.retriever.eval()
            self.context_encoder.eval()
            try:
                with torch.no_grad():
                    top_idx, top_sims, weights = self.retriever.retrieve(
                        q_sig, case_sigs, type_post_t,
                        k=k, temperature=temperature,
                    )
                    case_feats = self._gather_case_features(top_idx)   # (1, K, _)
                    context = self.context_encoder(case_feats, weights)   # (1, ctx_dim)
            finally:
                if was_training_retriever:
                    self.retriever.train()
                if was_training_context:
                    self.context_encoder.train()
        else:
            # Ablation: pure cosine similarity. Equivalent to setting
            # all L_τ to 0 (using only γ * cosine). We compute directly
            # rather than zeroing L_τ in-place (no side effects).
            was_training_retriever = self.retriever.training
            was_training_context = self.context_encoder.training
            self.retriever.eval()
            self.context_encoder.eval()
            try:
                with torch.no_grad():
                    cosine = q_sig @ case_sigs.T   # (1, N)
                    scores = cosine * self.retriever.gamma
                    n_avail = case_sigs.shape[0]
                    kk = min(k, n_avail)
                    top_sims, top_idx = scores.topk(kk, dim=-1)
                    weights = F.softmax(top_sims / temperature, dim=-1)
                    case_feats = self._gather_case_features(top_idx)
                    context = self.context_encoder(case_feats, weights)
            finally:
                if was_training_retriever:
                    self.retriever.train()
                if was_training_context:
                    self.context_encoder.train()

        # 4) package the result
        top_idx_np = top_idx[0].cpu().numpy()
        top_sims_np = top_sims[0].cpu().numpy().astype(np.float64)
        weights_np = weights[0].cpu().numpy().astype(np.float64)
        retrieved_ids = tuple(self._case_ids[int(i)] for i in top_idx_np)
        context_np = context[0].cpu().numpy().astype(np.float64)

        return RetrievalResult(
            case_ids=retrieved_ids,
            similarities=top_sims_np,
            weights=weights_np,
            type_posterior=type_post.copy(),
            query_signature=q_sig_np.astype(np.float64).copy(),
            context_vector=context_np,
            temperature=float(temperature),
        )

    def retrieve_with_attribution(
        self,
        query_trajectory: np.ndarray,
        type_posterior: np.ndarray,
        *,
        k: int = DEFAULT_K_RETRIEVAL,
        temperature: float = DEFAULT_RETRIEVAL_TEMPERATURE,
    ) -> tuple[RetrievalResult, dict[str, Any]]:
        """Same as ``retrieve`` but also returns a structured
        explainability report.

        The report contains:
        * ``"per_type_contribution"``: shape (K, len(CRISIS_TYPES));
          how much each crisis-type slot of the metric contributed
          to each retrieved case's score.
        * ``"cosine_baseline"``: shape (K,); the γ-cosine baseline
          component for each retrieved case (the part of the score
          NOT explained by type conditioning).
        * ``"crisis_types_retrieved"``: the crisis_type of each
          retrieved case, for sanity checking.
        """
        result = self.retrieve(
            query_trajectory, type_posterior,
            k=k, temperature=temperature, use_type_conditioning=True,
        )
        # Re-compute per-type contributions for the retrieved cases.
        # ``result.query_signature`` was marked read-only by
        # RetrievalResult.__post_init__; .copy() so torch.as_tensor
        # gets a writeable backing buffer.
        q_sig = torch.as_tensor(
            result.query_signature.copy(),
            dtype=torch.float32, device=self._device,
        ).unsqueeze(0)
        case_sigs = torch.as_tensor(
            self._signature_matrix, dtype=torch.float32, device=self._device,
        )
        # Indices of retrieved cases in the library
        retrieved_idx = np.array(
            [self._case_ids.index(cid) for cid in result.case_ids],
        )
        c_sigs_retrieved = case_sigs[retrieved_idx]   # (K, D)

        with torch.no_grad():
            # Per-type bilinear scores via einsum
            q_proj = torch.einsum(
                "bd,tdr->btr", q_sig, self.retriever.L,
            )   # (1, T, R)
            c_proj = torch.einsum(
                "kd,tdr->ktr", c_sigs_retrieved, self.retriever.L,
            )   # (K, T, R)
            per_type = torch.einsum(
                "btr,ktr->btk", q_proj, c_proj,
            )   # (1, T, K)
            cosine = (q_sig @ c_sigs_retrieved.T)[0] * self.retriever.gamma   # (K,)

        report = {
            "per_type_contribution": per_type[0].cpu().numpy().T,   # (K, T)
            "cosine_baseline": cosine.cpu().numpy(),                # (K,)
            "crisis_types": list(CRISIS_TYPES),
            "crisis_types_retrieved": [
                self.library[cid].crisis_type for cid in result.case_ids
            ],
        }
        return result, report

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(
        self,
        *,
        n_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 0.5,
        label_smoothing: float = 0.1,
        grad_clip: float = 1.0,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train the retriever (and fit the context encoder's
        standardisation).

        The signature encoder is held FROZEN. Phase C of the orchestrator
        co-trains the retriever and context encoder against the library's
        crisis-type labels via an InfoNCE-style loss: for each case as
        query, the loss rewards higher similarity to same-type cases and
        penalises higher similarity to different-type cases.

        Parameters
        ----------
        n_epochs : int, default 200
        lr, weight_decay : AdamW hyperparameters.
        temperature : float, default 0.5
            Temperature for the InfoNCE softmax during training.
            Distinct from the retrieval-time temperature.
        label_smoothing : float in [0, 1), default 0.1
            Smooths the one-hot type posterior used as the query's
            conditioning during training. With ``label_smoothing=0.1``,
            the conditioning becomes ``0.9 * one_hot + 0.1 / n_types``,
            which makes the retriever's metric work even for off-
            diagonal posteriors at inference time.
        grad_clip : float, default 1.0
            Global gradient-norm clip.
        verbose : bool, default False

        Returns
        -------
        history : dict
            Training history with keys ``"epoch"`` and ``"loss"``.
        """
        if self.n_cases < _MIN_LIBRARY_SIZE_FOR_FIT:
            raise CaseLibraryError(
                f"AnalogyEngine.fit: need >= {_MIN_LIBRARY_SIZE_FOR_FIT} "
                f"cases; got {self.n_cases}"
            )
        if not self.signature_encoder._is_fitted:
            raise RetrieverNotFittedError(
                "AnalogyEngine.fit: signature_encoder must be .fit first. "
                "Phase B (encoder) precedes Phase C (retriever)."
            )
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1); got {label_smoothing}"
            )

        # 1) fit context encoder's standardisation stats
        self.context_encoder.fit(self.library)

        # 2) train the retriever
        device = self._device
        case_sigs = torch.as_tensor(
            self._signature_matrix, dtype=torch.float32, device=device,
        )   # (N, D)
        labels = torch.as_tensor(
            self._crisis_type_labels, dtype=torch.long, device=device,
        )   # (N,)
        n_cases = case_sigs.shape[0]
        n_types = len(CRISIS_TYPES)

        # Build one-hot posteriors with smoothing
        eye = torch.eye(n_types, device=device, dtype=torch.float32)
        type_post = eye[labels]   # (N, T)
        if label_smoothing > 0:
            type_post = (
                (1 - label_smoothing) * type_post + label_smoothing / n_types
            )

        # Positive mask: same crisis type, excluding self
        pos_mask = (
            (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            - torch.eye(n_cases, device=device, dtype=torch.float32)
        ).clamp(min=0.0)
        eye_mask = torch.eye(n_cases, device=device, dtype=torch.bool)

        n_pos = pos_mask.sum(dim=-1)
        valid_mask = n_pos > 0
        if int(valid_mask.sum().item()) == 0:
            raise CaseLibraryError(
                "AnalogyEngine.fit: every case is the unique example of "
                "its crisis type; no positive pairs available for "
                "contrastive training. Add more cases or coarsen the "
                "crisis-type vocabulary."
            )

        self.retriever.to(device)
        self.retriever.train()
        optim = torch.optim.AdamW(
            self.retriever.parameters(), lr=lr, weight_decay=weight_decay,
        )

        history: dict[str, list[float]] = {"epoch": [], "loss": []}

        for epoch in range(n_epochs):
            scores = self.retriever(case_sigs, case_sigs, type_post)   # (N, N)
            # Mask self-similarity to -inf so it does not enter the softmax,
            # then divide by temperature.
            scores = scores.masked_fill(eye_mask, float("-inf"))
            scores = scores / temperature

            log_softmax = F.log_softmax(scores, dim=-1)
            # After log_softmax the diagonal is -inf. ``pos_mask`` is 0 on
            # the diagonal, so the product would be 0 * (-inf) = NaN under
            # PyTorch's IEEE semantics. Zero out the diagonal of
            # ``log_softmax`` *after* the softmax — mathematically a no-op
            # since pos_mask is zero there, but it prevents the NaN.
            log_softmax = log_softmax.masked_fill(eye_mask, 0.0)
            # Per-anchor InfoNCE: -mean over positives of log P(positive)
            per_anchor = -(pos_mask * log_softmax).sum(dim=-1) / n_pos.clamp_min(1)
            loss = per_anchor[valid_mask].mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.retriever.parameters(), max_norm=grad_clip,
                )
            optim.step()

            history["epoch"].append(epoch)
            history["loss"].append(float(loss.item()))
            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                logger.info(
                    "AnalogyEngine.fit: epoch %d/%d  infonce_loss=%.5f  "
                    "gamma=%.4f",
                    epoch + 1, n_epochs, history["loss"][-1],
                    float(self.retriever.gamma.item()),
                )

        self.retriever.eval()
        self.context_encoder.eval()
        self.retriever._is_fitted = True
        self._fit_history = history
        self._is_fitted = True
        return history

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> Path:
        """Save the engine to ``path`` (a directory).

        The signature encoder is NOT saved as part of the engine; it
        is the orchestrator's responsibility to save the encoder
        separately. This separation keeps the engine artefact focused
        and matches the Phase B / Phase C training boundary.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save retriever and context encoder
        self.retriever.save(path)
        self.context_encoder.save(path)

        # Save manifest
        current_lib_ck = self.library.library_checksum()
        if current_lib_ck != self._library_checksum_at_init:
            warnings.warn(
                f"AnalogyEngine.save: library checksum changed since "
                f"engine construction (init={self._library_checksum_at_init[:12]}, "
                f"current={current_lib_ck[:12]}). Recording the current "
                f"checksum; the engine should be re-fit on the new library.",
                stacklevel=2,
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "case_schema_version": CASE_SCHEMA_VERSION,
            "saved_at": _utc_now_iso(),
            "library_checksum": current_lib_ck,
            "n_cases": self.n_cases,
            "is_fitted": self._is_fitted,
            "fit_history_summary": (
                {
                    "n_epochs": len(self._fit_history.get("loss", [])),
                    "final_loss": (
                        self._fit_history["loss"][-1]
                        if self._fit_history.get("loss") else None
                    ),
                }
                if self._fit_history else {}
            ),
            "retriever_config": self.retriever.config,
            "context_encoder_config": self.context_encoder.config,
        }
        _atomic_write_text(
            path / _ENGINE_MANIFEST_NAME,
            json.dumps(manifest, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        logger.info(
            "AnalogyEngine.save: wrote engine to %s "
            "(library_checksum=%s..., fitted=%s)",
            path, current_lib_ck[:12], self._is_fitted,
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        library: CaseLibrary,
        signature_encoder: CaseSignatureEncoder,
        device: Union[str, torch.device] = "cpu",
        strict_checksum: bool = True,
    ) -> "AnalogyEngine":
        """Reconstruct an engine from disk.

        Parameters
        ----------
        path : Path
            Directory written by ``AnalogyEngine.save``.
        library : CaseLibrary
            The case library to bind the engine to. Its content
            checksum is compared to the saved one.
        signature_encoder : CaseSignatureEncoder
            The signature encoder. Not saved with the engine; the
            caller is responsible for loading it (typically via
            ``CaseSignatureEncoder.load``).
        device : torch device.
        strict_checksum : bool, default True
            If True, a library-checksum mismatch raises
            ``LibraryChecksumMismatchError``. If False, it warns.
        """
        path = Path(path)
        manifest_path = path / _ENGINE_MANIFEST_NAME
        if not manifest_path.is_file():
            raise AnalogyEngineError(
                f"AnalogyEngine.load: no manifest at {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _check_compatible_version(
            manifest.get("schema_version", SCHEMA_VERSION),
            context=f"AnalogyEngine@{path}",
        )

        # Library checksum check
        expected_ck = manifest.get("library_checksum")
        actual_ck = library.library_checksum()
        if expected_ck is not None and actual_ck != expected_ck:
            msg = (
                f"AnalogyEngine.load: library checksum mismatch "
                f"(expected {expected_ck[:12]}, got {actual_ck[:12]}). "
                f"The engine was trained against a different library "
                f"content. Either supply the original library or re-fit "
                f"the engine."
            )
            if strict_checksum:
                raise LibraryChecksumMismatchError(msg)
            warnings.warn(msg, stacklevel=2)

        retriever = ConditionalRetriever.load(path, map_location=device)
        context_encoder = CaseContextEncoder.load(path, map_location=device)

        engine = cls(
            library=library,
            signature_encoder=signature_encoder,
            retriever=retriever,
            context_encoder=context_encoder,
            device=device,
        )
        engine._is_fitted = bool(manifest.get("is_fitted", False))
        # Note: _fit_history is not persisted in full; we only restore
        # the summary. To recover the full history, re-fit.
        engine._fit_history = {}
        return engine

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, Any]:
        """Human-readable summary for logging or printing."""
        return {
            "schema_version": SCHEMA_VERSION,
            "library_checksum_at_init": self._library_checksum_at_init,
            "n_cases": self.n_cases,
            "retriever_params": int(
                sum(p.numel() for p in self.retriever.parameters())
            ),
            "context_encoder_params": int(
                sum(p.numel() for p in self.context_encoder.parameters())
            ),
            "is_fitted": self._is_fitted,
            "fit_history_n_epochs": len(self._fit_history.get("loss", [])),
            "final_fit_loss": (
                self._fit_history["loss"][-1]
                if self._fit_history.get("loss") else None
            ),
            "device": str(self._device),
        }

    def __repr__(self) -> str:
        return (
            f"AnalogyEngine(n_cases={self.n_cases}, "
            f"signature_dim={self.retriever.config['signature_dim']}, "
            f"n_types={self.retriever.config['n_types']}, "
            f"rank={self.retriever.config['rank']}, "
            f"fitted={self._is_fitted})"
        )


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def build_retrieval_index(
    library: CaseLibrary,
    signature_encoder: CaseSignatureEncoder,
) -> dict[str, Any]:
    """Convenience: build a dict of precomputed retrieval index matrices
    without going through the full AnalogyEngine. Useful for unit tests
    and for one-off retrieval experiments."""
    case_ids, sig_mat = build_signature_matrix(library, signature_encoder)
    if len(case_ids) == 0:
        return {
            "case_ids": (),
            "signature_matrix": np.empty((0, 0), dtype=np.float32),
            "policy_fingerprint_matrix": np.empty(
                (0, POLICY_FINGERPRINT_DIM), dtype=np.float64,
            ),
            "outcome_fingerprint_matrix": np.empty(
                (0, OUTCOME_FINGERPRINT_DIM), dtype=np.float64,
            ),
            "crisis_type_labels": np.empty(0, dtype=np.int64),
        }
    return {
        "case_ids": case_ids,
        "signature_matrix": sig_mat.astype(np.float32, copy=False),
        "policy_fingerprint_matrix": np.stack([
            compute_policy_fingerprint(library[cid]) for cid in case_ids
        ], axis=0),
        "outcome_fingerprint_matrix": np.stack([
            compute_outcome_fingerprint(library[cid]) for cid in case_ids
        ], axis=0),
        "crisis_type_labels": np.array(
            [CRISIS_TYPES.index(library[cid].crisis_type) for cid in case_ids],
            dtype=np.int64,
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> int:
    """CLI: ``python -m em_fin_stability.analogy_engine <command> [args]``.

    Commands
    --------
    summary <engine_path> <library_path> <encoder_path>
        Load an engine + library + encoder and print a JSON summary.
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m em_fin_stability.analogy_engine",
        description="Inspect a saved Analogy Engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_sum = sub.add_parser("summary", help="Print engine summary.")
    p_sum.add_argument("engine_path", type=Path)
    p_sum.add_argument("library_path", type=Path)
    p_sum.add_argument("encoder_path", type=Path)
    p_sum.add_argument(
        "--no-strict", action="store_true",
        help="Allow library-checksum mismatch (warn instead of error).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _configure_logging(args.log_level)

    try:
        if args.command == "summary":
            lib = CaseLibrary.load(args.library_path, verify_checksums=True)
            enc = CaseSignatureEncoder.load(args.encoder_path)
            engine = AnalogyEngine.load(
                args.engine_path, library=lib, signature_encoder=enc,
                strict_checksum=not args.no_strict,
            )
            print(json.dumps(engine.summary(), indent=2, sort_keys=True))
            return 0
        else:  # pragma: no cover
            parser.print_help()
            return 2
    except CaseMemoryError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
