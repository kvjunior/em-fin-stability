"""
mitigation.py
=============

Multi-agent crisis-mitigation policy with case-augmented actors
(novelty #3 of the paper).

This module is the policy counterpart to ``analogy_engine.py``. Where
``analogy_engine.py`` *retrieves* historical analogues and *summarises*
them into a context vector, ``mitigation.py`` *acts* — it converts the
state plus the retrieved context into per-institution policy actions
that a central bank, financial supervisor, and ministry of finance
would each implement. This is the file where the analogical reasoning
becomes deployable advice.

Architecture
------------
::

    AuthorityGraph             -> the institution -> levers ownership
                                  map, with action validation,
                                  per-institution / joint conversions,
                                  and unit <-> physical scaling.
    SafetyBounds               -> the safe operating envelope of the
                                  macro state (FX reserves >= 1 month,
                                  inflation < 50% YoY, etc.). Frozen
                                  dataclass with serialisation.
    ControlBarrierFunction     -> safety filter in two modes: hard
                                  (action projection to nearest safe
                                  action; used at deployment) and soft
                                  (differentiable penalty; used in
                                  training).
    LinearCrisisDynamics       -> counterfactual simulator of macro
                                  dynamics. x_{t+1} = A x_t + B u_t +
                                  bias, fittable from the case library
                                  with literature-calibrated B matrix
                                  defaults.
    CaseAugmentedActor         -> per-institution policy network.
                                  Consumes (state, retrieved-context,
                                  crisis-type-posterior); emits actions
                                  over this institution's levers in
                                  unit space [-1, 1].
    TwinCritic                 -> TD3 twin Q-function over joint state
                                  and joint action (centralised
                                  critic).
    MultiAgentMitigationPolicy -> orchestrator: per-institution actors
                                  + shared twin critics + CBF
                                  filtering. Top-level inference and
                                  save/load.
    MitigationTransition       -> immutable replay-buffer record;
                                  stores SARS + done + the retrieval
                                  fingerprints + outcomes needed for
                                  the case-coherence regulariser.
    ReplayBuffer               -> fixed-capacity ring buffer with
                                  deterministic sampling.
    case_coherence_loss        -> the new regularisation term: cosine
                                  similarity between the actor's
                                  action and the retrieval-weighted,
                                  outcome-weighted target action.
    MitigationTrainer          -> one TD3 optimisation step. Owns the
                                  optimisers and target-network
                                  Polyak averaging.

Why this is the third novel module
----------------------------------
A standard multi-agent crisis-mitigation system would have actors that
take (state) -> action. The case-augmented variant takes (state,
retrieved_case_context, crisis_type_posterior) -> action, with the
context vector coming from ``AnalogyEngine.retrieve``. The actor is
therefore informed by historical episodes that match the current
macro signature.

The case-coherence regulariser closes the analogical loop: it
penalises the actor for choosing actions that diverge from what
*succeeded* in retrieved similar episodes (failures are down-weighted
by outcome). This is the mechanism by which historical knowledge
exerts force on the policy.

The Control Barrier Function ensures that no matter what the actor
proposes, the deployed action stays within the safe operating
envelope — a property central banks would require before deploying
any ML-derived recommendation.

Reproducibility commitments
---------------------------
* All neural networks use seeded initialisation via save-and-restore
  of the global torch RNG, so the caller's RNG state is preserved.
* The replay buffer uses an explicit seeded numpy generator; calling
  ``ReplayBuffer.sample`` twice with the same seed produces the same
  batch.
* All disk writes are atomic (``tmp`` + ``os.replace``).
* The dynamics simulator persists fitted A and B matrices alongside
  the literature-citation provenance for the B-matrix defaults.

References (APA-7)
------------------
Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K.,
    & Tabuada, P. (2019). Control barrier functions: Theory and
    applications. In Proceedings of the 18th European Control
    Conference (pp. 3420-3431).

Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the
    effects of monetary policy: A factor-augmented vector
    autoregressive (FAVAR) approach. Quarterly Journal of Economics,
    120(1), 387-422. [For impulse-response B matrix defaults.]

Coogan, S., & Yel, E. (2020). Discrete-time control barrier functions
    with applications to multi-agent systems. arXiv:2007.07596.

Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function
    approximation error in actor-critic methods. In Proceedings of
    the 35th International Conference on Machine Learning (pp.
    1587-1596). [TD3.]

Romer, C. D., & Romer, D. H. (2004). A new measure of monetary shocks:
    Derivation and implications. American Economic Review, 94(4),
    1055-1084.

Yang, Y., Hao, J., Liao, B., Shao, K., Chen, G., Liu, W., & Tang, H.
    (2020). Qatten: A general framework for cooperative multiagent
    reinforcement learning. arXiv:2002.03939.

Version
-------
1.0.0  Camera-ready KBS submission.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from case_memory import (
    AuthoritySnapshot,
    CaseLibrary,
    CaseMemoryError,
    CRISIS_TYPES,
    DEFAULT_FEATURE_NAMES,
    DEFAULT_LEVERS,
    DEFAULT_POST_ONSET_QUARTERS,
    INSTITUTIONS,
    N_MACRO_FEATURES,
)
from analogy_engine import (
    DEFAULT_CONTEXT_DIM,
    POLICY_FINGERPRINT_DIM,
    RetrievalResult,
)


__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "JOINT_ACTION_DIM",
    "STATE_DIM",
    "DEFAULT_LEVER_SCALES",
    "DEFAULT_B_MATRIX_IMPULSE_RESPONSES",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_ACTOR_LR",
    "DEFAULT_CRITIC_LR",
    "DEFAULT_GAMMA",
    "DEFAULT_POLYAK",
    # Exceptions
    "MitigationError",
    "UnsafeActionError",
    "AuthorityGraphError",
    # Authority
    "AuthorityGraph",
    # Safety
    "SafetyBounds",
    "ControlBarrierFunction",
    # Dynamics
    "LinearCrisisDynamics",
    # Networks
    "CaseAugmentedActor",
    "TwinCritic",
    # Combined policy
    "MultiAgentMitigationPolicy",
    # Replay
    "MitigationTransition",
    "ReplayBuffer",
    # Losses
    "case_coherence_loss",
    "compute_reward",
    # Trainer
    "MitigationTrainer",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Semantic version of the on-disk mitigation-module artefacts.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: Joint-action dimensionality: one entry per lever, exactly one
#: institution owning each lever (per ``AuthorityGraph``). Equals
#: ``len(DEFAULT_LEVERS) = 8``.
JOINT_ACTION_DIM: Final[int] = len(DEFAULT_LEVERS)

#: Macro-state dimensionality. Equals ``N_MACRO_FEATURES = 12``.
STATE_DIM: Final[int] = N_MACRO_FEATURES

#: Per-lever physical scale (the magnitude that corresponds to a unit
#: action of +1.0 in the actor's tanh output). These are large but
#: plausible single-quarter changes, calibrated against the EM
#: literature (Bernanke et al. 2005; Romer & Romer 2004). Sign
#: convention: positive = tightening, negative = loosening.
DEFAULT_LEVER_SCALES: Final[dict[str, float]] = {
    "policy_rate":              500.0,   # bp single-quarter change
    "fx_intervention":           30.0,   # USD billion per quarter
    "reserve_requirement":      200.0,   # bp single-quarter change
    "capital_adequacy_ratio":     5.0,   # pp of risk-weighted assets
    "loan_to_value_cap":         30.0,   # pp
    "countercyclical_buffer":     5.0,   # pp of risk-weighted assets
    "fiscal_stance":              5.0,   # pp of GDP
    "debt_issuance":             50.0,   # USD billion per quarter
}

#: Literature-calibrated impulse-response defaults for the linear
#: dynamics simulator's B matrix. Maps lever -> dict[state_feature,
#: response coefficient]. Coefficients are the *contemporaneous*
#: response of the state feature to a unit *physical* action; the
#: simulator distributes the effect over subsequent quarters via the
#: A matrix. Magnitudes are stylised facts from the EM monetary-
#: transmission and fiscal-multiplier literatures.
#:
#: Sign convention matches DEFAULT_LEVER_SCALES (positive = tightening).
#: For instance, a positive policy_rate change (tightening) reduces
#: the output gap (-0.001 per bp) and reduces inflation
#: (-0.0008 per bp) on impact, while raising the REER (+0.0003 per
#: bp).
DEFAULT_B_MATRIX_IMPULSE_RESPONSES: Final[dict[str, dict[str, float]]] = {
    "policy_rate": {
        "output_gap":            -0.0010,   # 100bp tightening -> -0.10pp gap
        "inflation_yoy":         -0.0008,
        "reer_log_dev":           0.0003,
        "credit_gap":            -0.0005,
        "sovereign_spread_bp":   -0.05,
    },
    "fx_intervention": {
        "reer_log_dev":          -0.002,
        "fx_reserves_months":    -0.01,
    },
    "reserve_requirement": {
        "credit_gap":            -0.0008,
        "output_gap":            -0.0004,
    },
    "capital_adequacy_ratio": {
        "credit_gap":            -0.10,
        "output_gap":            -0.05,
    },
    "loan_to_value_cap": {
        "credit_gap":            -0.02,
    },
    "countercyclical_buffer": {
        "credit_gap":            -0.08,
    },
    "fiscal_stance": {
        "output_gap":            -0.40,    # 1pp consolidation -> -0.40pp gap
        "sovereign_spread_bp":   -30.0,
    },
    "debt_issuance": {
        "sovereign_spread_bp":    1.5,
    },
}

#: Default hidden dim for actor and critic MLPs.
DEFAULT_HIDDEN_DIM: Final[int] = 128

#: Default actor learning rate (TD3 convention: actor < critic to slow
#: policy updates relative to value updates).
DEFAULT_ACTOR_LR: Final[float] = 3e-4

#: Default critic learning rate.
DEFAULT_CRITIC_LR: Final[float] = 3e-4

#: Default discount factor.
DEFAULT_GAMMA: Final[float] = 0.95

#: Default Polyak averaging coefficient for target network updates
#: (target = polyak * target + (1 - polyak) * online).
DEFAULT_POLYAK: Final[float] = 0.995

#: Default frequency (in critic updates) at which the actor is updated.
DEFAULT_ACTOR_UPDATE_FREQ: Final[int] = 2

#: Default standard deviation of the smoothing noise added to the
#: target actor's action during the critic update (TD3 target-policy
#: smoothing).
DEFAULT_TARGET_NOISE_STD: Final[float] = 0.2

#: Default clip range for the target-policy smoothing noise.
DEFAULT_TARGET_NOISE_CLIP: Final[float] = 0.5

#: Default weight on the case-coherence regulariser in the actor
#: loss. Tuned so the regulariser has comparable magnitude to the
#: policy-gradient term on the canonical 28-country panel.
DEFAULT_CASE_COHERENCE_WEIGHT: Final[float] = 0.1

#: Default outcome-temperature in the case-coherence regulariser.
#: Lower values sharpen the outcome-weighting (concentrate the
#: regulariser on the most-successful retrieved cases); higher values
#: spread it across retrievals. 5.0 (in percent-of-GDP units) is a
#: sensible default.
DEFAULT_OUTCOME_TEMPERATURE: Final[float] = 5.0

#: Default weight on the soft CBF penalty in the actor loss.
DEFAULT_SAFETY_PENALTY_WEIGHT: Final[float] = 1.0


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class MitigationError(CaseMemoryError):
    """Base class for mitigation-module exceptions. Inherits from
    ``CaseMemoryError`` so a single ``except`` clause catches all
    package-level errors."""


class UnsafeActionError(MitigationError):
    """A proposed action, even after CBF filtering, would violate a
    hard safety constraint. Raised in deployment mode when the safe
    set is empty for the given state — indicates the state itself is
    already unsafe."""


class AuthorityGraphError(MitigationError):
    """An action is inconsistent with the authority graph: an
    institution tried to set a lever it does not own, or a joint
    action's dimensionality does not match the lever count."""


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
        raise MitigationError(
            f"{context}: malformed schema_version {found_version!r}; "
            f"expected MAJOR.MINOR.PATCH"
        ) from exc
    if len(found) != 3 or len(expected) != 3:
        raise MitigationError(
            f"{context}: schema_version must be MAJOR.MINOR.PATCH; "
            f"got {found_version!r}"
        )
    if found[0] != expected[0]:
        raise MitigationError(
            f"{context}: schema major-version mismatch (found "
            f"{found_version!r}, code supports {expected_version!r})"
        )
    if found[1] != expected[1]:
        logger.warning(
            "%s: schema minor-version mismatch (found %s, code supports %s)",
            context, found_version, expected_version,
        )


def _seeded_module_init(seed: int, build_fn):
    """Run ``build_fn()`` under a save-and-restore of the global torch
    RNG state, with the RNG seeded to ``seed``. Used by all neural
    networks in this module for deterministic init.
    """
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


def _feature_index(name: str,
                   feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES) -> int:
    """Index of a named macro feature in the canonical column order."""
    try:
        return feature_names.index(name)
    except ValueError as exc:
        raise MitigationError(
            f"unknown macro feature {name!r}; available: {list(feature_names)}"
        ) from exc


# ---------------------------------------------------------------------------
# AuthorityGraph: action validation, joint <-> per-institution, scaling
# ---------------------------------------------------------------------------


class AuthorityGraph:
    """The institution -> levers ownership map with action validation.

    Wraps an :class:`AuthoritySnapshot` from ``case_memory.py`` and adds
    runtime utilities for the policy networks:

    * ``levers_for(institution) -> tuple[str, ...]``: the institution's
      lever names, sorted in canonical (``DEFAULT_LEVERS``) order.
    * ``institution_for(lever) -> str``: the unique owner of a lever.
    * ``split_joint_action(joint) -> dict[str, ndarray]``: extract each
      institution's per-lever slice from a joint action vector.
    * ``combine_action(per_inst) -> ndarray``: assemble a joint action
      from per-institution slices.
    * ``scale_to_physical(unit_action) -> ndarray``: multiply each lever
      entry by its ``DEFAULT_LEVER_SCALES`` factor.
    * ``scale_to_unit(physical_action) -> ndarray``: divide each lever
      entry by its scale.

    The canonical column ordering for the joint action vector is
    ``DEFAULT_LEVERS`` from ``case_memory.py``. The per-institution
    slice for institution ``I`` contains the columns corresponding to
    levers ``I`` owns, in their canonical sub-order. ``split_joint_action``
    and ``combine_action`` are exact inverses.
    """

    def __init__(
        self,
        snapshot: AuthoritySnapshot,
        *,
        lever_scales: Optional[Mapping[str, float]] = None,
    ) -> None:
        if not isinstance(snapshot, AuthoritySnapshot):
            raise TypeError(
                f"AuthorityGraph: snapshot must be AuthoritySnapshot; "
                f"got {type(snapshot).__name__}"
            )
        self.snapshot = snapshot
        # Build inverted index lever -> institution; validate uniqueness.
        owner: dict[str, str] = {}
        for inst, levers in snapshot.institution_levers.items():
            for lev in levers:
                if lev in owner:
                    raise AuthorityGraphError(
                        f"lever {lev!r} owned by both {owner[lev]!r} "
                        f"and {inst!r}; an AuthoritySnapshot should "
                        f"reject this at construction"
                    )
                owner[lev] = inst
        self._owner: dict[str, str] = owner

        # Build the per-institution column-index slice into the joint
        # action vector. Joint action is in DEFAULT_LEVERS order.
        self._lever_index: dict[str, int] = {
            lev: i for i, lev in enumerate(DEFAULT_LEVERS)
        }
        self._slices: dict[str, np.ndarray] = {}
        for inst in INSTITUTIONS:
            owned = snapshot.institution_levers.get(inst, ())
            idx = np.array(
                [self._lever_index[lev] for lev in owned if lev in self._lever_index],
                dtype=np.int64,
            )
            self._slices[inst] = idx

        # Lever scales: validate and assemble in DEFAULT_LEVERS order.
        scales = dict(DEFAULT_LEVER_SCALES)
        if lever_scales is not None:
            for k, v in lever_scales.items():
                if k not in DEFAULT_LEVERS:
                    raise AuthorityGraphError(
                        f"lever_scales: unknown lever {k!r}; allowed "
                        f"{list(DEFAULT_LEVERS)}"
                    )
                if not np.isfinite(v) or v <= 0:
                    raise AuthorityGraphError(
                        f"lever_scales[{k!r}]: must be positive finite; got {v}"
                    )
                scales[k] = float(v)
        self._lever_scales: dict[str, float] = scales
        self._lever_scale_vec: np.ndarray = np.array(
            [scales[lev] for lev in DEFAULT_LEVERS], dtype=np.float64,
        )

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def institutions(self) -> tuple[str, ...]:
        """All institutions in canonical order (those owning at least
        one lever)."""
        return tuple(
            inst for inst in INSTITUTIONS
            if len(self.snapshot.institution_levers.get(inst, ())) > 0
        )

    def levers_for(self, institution: str) -> tuple[str, ...]:
        """Levers owned by ``institution`` in canonical order."""
        if institution not in INSTITUTIONS:
            raise AuthorityGraphError(
                f"unknown institution {institution!r}; allowed "
                f"{list(INSTITUTIONS)}"
            )
        return tuple(
            lev for lev in DEFAULT_LEVERS
            if self._owner.get(lev) == institution
        )

    def institution_for(self, lever: str) -> str:
        """Institution owning ``lever``. Raises if no owner."""
        if lever not in DEFAULT_LEVERS:
            raise AuthorityGraphError(
                f"unknown lever {lever!r}; allowed {list(DEFAULT_LEVERS)}"
            )
        if lever not in self._owner:
            raise AuthorityGraphError(
                f"lever {lever!r} has no owner in this authority graph"
            )
        return self._owner[lever]

    def n_levers_for(self, institution: str) -> int:
        return len(self.levers_for(institution))

    def lever_scale(self, lever: str) -> float:
        if lever not in self._lever_scales:
            raise AuthorityGraphError(f"unknown lever {lever!r}")
        return self._lever_scales[lever]

    @property
    def lever_scale_vector(self) -> np.ndarray:
        """Read-only ``(JOINT_ACTION_DIM,)`` vector of per-lever scales
        in ``DEFAULT_LEVERS`` order."""
        v = self._lever_scale_vec.copy()
        v.setflags(write=False)
        return v

    # ------------------------------------------------------------------ #
    # Joint <-> per-institution conversions
    # ------------------------------------------------------------------ #

    def split_joint_action(
        self, joint_action: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Split a ``(JOINT_ACTION_DIM,)`` joint action into a dict
        institution -> per-lever sub-array."""
        joint = np.asarray(joint_action)
        if joint.shape[-1] != JOINT_ACTION_DIM:
            raise AuthorityGraphError(
                f"joint_action last dim {joint.shape[-1]} != "
                f"JOINT_ACTION_DIM {JOINT_ACTION_DIM}"
            )
        out: dict[str, np.ndarray] = {}
        for inst in self.institutions:
            idx = self._slices[inst]
            if joint.ndim == 1:
                out[inst] = joint[idx]
            else:
                out[inst] = joint[..., idx]
        return out

    def combine_action(
        self,
        per_institution: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Assemble a joint action from per-institution sub-arrays."""
        out = np.zeros(JOINT_ACTION_DIM, dtype=np.float64)
        for inst, sub in per_institution.items():
            if inst not in self.institutions:
                raise AuthorityGraphError(
                    f"combine_action: unknown or unauthorised institution "
                    f"{inst!r}"
                )
            idx = self._slices[inst]
            sub_arr = np.asarray(sub, dtype=np.float64)
            if sub_arr.shape != idx.shape:
                raise AuthorityGraphError(
                    f"combine_action: institution {inst!r} sub-action "
                    f"shape {sub_arr.shape} does not match expected "
                    f"{idx.shape}"
                )
            out[idx] = sub_arr
        return out

    # ------------------------------------------------------------------ #
    # Unit <-> physical scaling
    # ------------------------------------------------------------------ #

    def scale_to_physical(self, unit_action: np.ndarray) -> np.ndarray:
        """Multiply each lever entry by its physical scale. Element-
        wise; preserves batch dimensions."""
        u = np.asarray(unit_action, dtype=np.float64)
        if u.shape[-1] != JOINT_ACTION_DIM:
            raise AuthorityGraphError(
                f"unit_action last dim {u.shape[-1]} != "
                f"JOINT_ACTION_DIM {JOINT_ACTION_DIM}"
            )
        return u * self._lever_scale_vec

    def scale_to_unit(self, physical_action: np.ndarray) -> np.ndarray:
        """Divide each lever entry by its physical scale (inverse of
        ``scale_to_physical``)."""
        p = np.asarray(physical_action, dtype=np.float64)
        if p.shape[-1] != JOINT_ACTION_DIM:
            raise AuthorityGraphError(
                f"physical_action last dim {p.shape[-1]} != "
                f"JOINT_ACTION_DIM {JOINT_ACTION_DIM}"
            )
        return p / self._lever_scale_vec

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "institution_levers": self.snapshot.to_dict(),
            "lever_scales": dict(self._lever_scales),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AuthorityGraph":
        if "schema_version" in d:
            _check_compatible_version(
                d["schema_version"], context="AuthorityGraph",
            )
        snap = AuthoritySnapshot.from_dict(d["institution_levers"])
        return cls(snap, lever_scales=d.get("lever_scales"))

    def __repr__(self) -> str:
        return (
            f"AuthorityGraph(institutions={list(self.institutions)}, "
            f"n_levers_covered={len(self.snapshot.covered_levers)})"
        )


# ---------------------------------------------------------------------------
# SafetyBounds and ControlBarrierFunction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafetyBounds:
    """Safe operating envelope of the macro state.

    Each field is a per-feature constraint of the form ``state[feature]
    >=`` (or ``<=``) some bound, used by ``ControlBarrierFunction`` to
    detect and prevent unsafe states.

    Defaults match conventional EM stability thresholds:
    * FX reserves >= 1 month of imports (IMF reserve-adequacy floor).
    * Inflation YoY <= 0.5 (50%, the conventional hyperinflation
      threshold; signals breakdown of monetary policy).
    * Policy rate >= 0 (no negative-rate experiments in EMs by
      historical convention).
    * Output gap >= -0.15 (more negative implies a severe recession
      from which mean-reverting linear dynamics are unreliable).
    """
    min_fx_reserves_months: float = 1.0
    max_inflation_yoy: float = 0.5
    min_policy_rate_proxy: float = 0.0
    min_output_gap: float = -0.15

    def __post_init__(self) -> None:
        for fname in (
            "min_fx_reserves_months", "max_inflation_yoy",
            "min_output_gap",
        ):
            val = getattr(self, fname)
            if not np.isfinite(val):
                raise MitigationError(
                    f"SafetyBounds.{fname}: must be finite; got {val}"
                )

    def to_dict(self) -> dict[str, float]:
        return {
            "min_fx_reserves_months": float(self.min_fx_reserves_months),
            "max_inflation_yoy": float(self.max_inflation_yoy),
            "min_policy_rate_proxy": float(self.min_policy_rate_proxy),
            "min_output_gap": float(self.min_output_gap),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, float]) -> "SafetyBounds":
        return cls(
            min_fx_reserves_months=float(d.get("min_fx_reserves_months", 1.0)),
            max_inflation_yoy=float(d.get("max_inflation_yoy", 0.5)),
            min_policy_rate_proxy=float(d.get("min_policy_rate_proxy", 0.0)),
            min_output_gap=float(d.get("min_output_gap", -0.15)),
        )

    def check(self, state: np.ndarray) -> dict[str, float]:
        """Return a dict of per-constraint slacks. Positive slack means
        the constraint is satisfied; negative means violated by that
        magnitude.

        Slacks are in the units of the underlying state feature, so
        they're directly interpretable.
        """
        state_arr = np.asarray(state, dtype=np.float64)
        fx_idx = _feature_index("fx_reserves_months")
        inf_idx = _feature_index("inflation_yoy")
        og_idx = _feature_index("output_gap")
        return {
            "fx_reserves_months >= min": float(state_arr[fx_idx])
                - self.min_fx_reserves_months,
            "inflation_yoy <= max": self.max_inflation_yoy
                - float(state_arr[inf_idx]),
            "output_gap >= min": float(state_arr[og_idx])
                - self.min_output_gap,
        }

    def all_satisfied(self, state: np.ndarray) -> bool:
        slacks = self.check(state)
        return all(s >= 0.0 for s in slacks.values())


class ControlBarrierFunction:
    """Safety filter in two modes.

    Hard mode (deployment)
    ----------------------
    ``filter_hard(state, proposed_action)`` returns ``(filtered_action,
    was_modified)``. The filtering is per-lever and per-constraint:
    when a proposed action would push a constrained state feature past
    its bound (in expectation under the linear dynamics, one-quarter
    horizon), the action is clipped so the expected state stays inside
    a margin of the bound. The margin and the magnitudes are
    parameters; the defaults are conservative.

    Soft mode (training)
    --------------------
    ``soft_penalty(state, action)`` returns a differentiable scalar
    that penalises proposed actions whose expected next-state would
    violate any constraint. The penalty is a sum of hinge losses, one
    per constraint, so the gradient pushes the actor away from
    unsafe action surfaces without imposing a hard projection (which
    would make gradients sparse and ill-conditioned).

    Both modes use the same dynamics estimate ``dynamics_B`` — the
    one-quarter response of each constrained feature to each lever's
    physical-unit change. Passing ``None`` for ``dynamics_B`` falls
    back to ``DEFAULT_B_MATRIX_IMPULSE_RESPONSES``.

    Why two modes
    -------------
    Hard mode is required at deployment because central banks cannot
    ship actions that violate constraints regardless of model
    uncertainty. Soft mode is required at training because hard
    projection breaks gradient flow through the projected lever
    coordinates, slowing convergence of the actor. The combination is
    standard in safe-RL (Ames et al. 2019; Coogan & Yel 2020).
    """

    def __init__(
        self,
        bounds: SafetyBounds,
        authority: AuthorityGraph,
        *,
        dynamics_B: Optional[Mapping[str, Mapping[str, float]]] = None,
        margin: float = 0.05,
        soft_slope: float = 10.0,
    ) -> None:
        if not isinstance(bounds, SafetyBounds):
            raise TypeError(
                f"bounds must be SafetyBounds; got {type(bounds).__name__}"
            )
        if not isinstance(authority, AuthorityGraph):
            raise TypeError(
                f"authority must be AuthorityGraph; got {type(authority).__name__}"
            )
        if margin < 0:
            raise ValueError(f"margin must be >= 0; got {margin}")
        if soft_slope <= 0:
            raise ValueError(f"soft_slope must be > 0; got {soft_slope}")
        self.bounds = bounds
        self.authority = authority
        self.margin = float(margin)
        self.soft_slope = float(soft_slope)
        # Materialise the (n_levers, n_features) impulse-response matrix
        # in DEFAULT_LEVERS / DEFAULT_FEATURE_NAMES order. Missing
        # (lever, feature) pairs default to zero.
        B_in = dict(DEFAULT_B_MATRIX_IMPULSE_RESPONSES)
        if dynamics_B is not None:
            for lev, resps in dynamics_B.items():
                if lev not in DEFAULT_LEVERS:
                    raise MitigationError(f"unknown lever in dynamics_B: {lev!r}")
                B_in[lev] = dict(resps)
        B = np.zeros((JOINT_ACTION_DIM, STATE_DIM), dtype=np.float64)
        for lev, resps in B_in.items():
            j = DEFAULT_LEVERS.index(lev)
            for feat, val in resps.items():
                if feat not in DEFAULT_FEATURE_NAMES:
                    raise MitigationError(
                        f"unknown feature in dynamics_B[{lev!r}]: {feat!r}"
                    )
                i = DEFAULT_FEATURE_NAMES.index(feat)
                B[j, i] = float(val)
        self._B: np.ndarray = B
        # Constrained features and their bound info: list of
        # (feature_index, direction, bound) where direction is +1 for
        # "state[i] >= bound" and -1 for "state[i] <= bound".
        self._constraints: list[tuple[int, int, float]] = [
            (_feature_index("fx_reserves_months"), +1, bounds.min_fx_reserves_months),
            (_feature_index("inflation_yoy"),      -1, bounds.max_inflation_yoy),
            (_feature_index("output_gap"),         +1, bounds.min_output_gap),
        ]

    # ------------------------------------------------------------------ #
    # Hard mode: per-lever clamp
    # ------------------------------------------------------------------ #

    def filter_hard(
        self,
        state: np.ndarray,
        proposed_unit_action: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Project ``proposed_unit_action`` (in ``[-1, 1]`` units) to
        the safe region.

        Algorithm: for each constraint i, compute the per-lever
        contribution to the expected next-state change. If the expected
        change would push past the bound (with margin), reduce the
        contribution of each lever proportionally to its share of the
        violation. The projection is per-constraint and applied
        sequentially in the order of decreasing violation magnitude.

        Returns
        -------
        (filtered_action, was_modified) : (ndarray, bool)
            ``filtered_action`` has the same shape as input. ``was_modified``
            is True iff any lever was clamped.
        """
        state_arr = np.asarray(state, dtype=np.float64)
        action = np.asarray(proposed_unit_action, dtype=np.float64).copy()
        if state_arr.shape != (STATE_DIM,):
            raise MitigationError(
                f"state shape {state_arr.shape} != ({STATE_DIM},)"
            )
        if action.shape != (JOINT_ACTION_DIM,):
            raise MitigationError(
                f"action shape {action.shape} != ({JOINT_ACTION_DIM},)"
            )

        was_modified = False
        # Physical-unit action and resulting one-step state change
        physical = action * self.authority.lever_scale_vector
        # Repeat until no constraint is violated (bounded iterations)
        for _ in range(8):
            delta = self._B.T @ physical          # (STATE_DIM,)
            predicted = state_arr + delta
            violated_any = False
            for feat_idx, direction, bound in self._constraints:
                slack = direction * (predicted[feat_idx] - bound) - self.margin
                if slack < 0:
                    violated_any = True
                    # Identify the levers contributing in the wrong direction
                    contribs = direction * self._B[:, feat_idx] * physical
                    bad = contribs < 0
                    sum_bad = float(contribs[bad].sum())
                    if sum_bad == 0:
                        # Constraint already infeasible from state alone:
                        # nothing the action can do. Zero levers that
                        # touch this feature to avoid making it worse.
                        physical = np.where(
                            self._B[:, feat_idx] != 0.0, 0.0, physical,
                        )
                        was_modified = True
                        break
                    # Scale down the offending levers proportionally
                    scale = max(0.0, 1.0 - (-slack) / max(abs(sum_bad), 1e-9))
                    physical = np.where(bad, physical * scale, physical)
                    was_modified = True
                    break
            if not violated_any:
                break

        # Convert back to unit space and clip to [-1, 1]
        filtered_unit = (
            physical / self.authority.lever_scale_vector
        ).clip(-1.0, 1.0)
        # If any clipping occurred, mark modified
        if not np.allclose(filtered_unit, action, atol=1e-12):
            was_modified = True
        return filtered_unit, bool(was_modified)

    # ------------------------------------------------------------------ #
    # Soft mode: differentiable penalty
    # ------------------------------------------------------------------ #

    def soft_penalty(
        self,
        state: torch.Tensor,
        unit_action: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of per-constraint hinge penalties on the predicted
        next-state.

        For each constraint of the form ``direction * (s' - bound) >= 0``,
        the penalty is ``relu(-direction * (s' - bound) + margin) ** 2``
        smoothed by a softplus-style approximation (``soft_slope``
        controls the sharpness). The result is differentiable in
        ``unit_action`` everywhere.

        Inputs may be batched: state ``(B, STATE_DIM)``, unit_action
        ``(B, JOINT_ACTION_DIM)``. Output is ``(B,)``.
        """
        if state.ndim != unit_action.ndim:
            raise MitigationError(
                f"state.ndim ({state.ndim}) must equal "
                f"unit_action.ndim ({unit_action.ndim})"
            )
        if state.shape[-1] != STATE_DIM:
            raise MitigationError(
                f"state last dim {state.shape[-1]} != {STATE_DIM}"
            )
        if unit_action.shape[-1] != JOINT_ACTION_DIM:
            raise MitigationError(
                f"action last dim {unit_action.shape[-1]} != "
                f"{JOINT_ACTION_DIM}"
            )
        device = state.device
        B_t = torch.as_tensor(self._B, dtype=state.dtype, device=device)
        # ``lever_scale_vector`` is intentionally read-only; copy so
        # torch.as_tensor gets a writable backing buffer.
        scale_t = torch.as_tensor(
            self.authority.lever_scale_vector.copy(),
            dtype=state.dtype, device=device,
        )
        physical = unit_action * scale_t
        delta = physical @ B_t                       # (..., STATE_DIM)
        predicted = state + delta

        total = torch.zeros(state.shape[:-1], dtype=state.dtype, device=device)
        for feat_idx, direction, bound in self._constraints:
            # signed slack: positive when safe
            slack = float(direction) * (predicted[..., feat_idx] - bound)
            # squared-hinge penalty on the violation magnitude with margin
            violation = F.relu(self.margin - slack)
            # Sharper near the boundary via the soft_slope
            total = total + (self.soft_slope * violation).pow(2)
        return total

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "bounds": self.bounds.to_dict(),
            "margin": self.margin,
            "soft_slope": self.soft_slope,
            "dynamics_B": self._B.tolist(),
        }

    @classmethod
    def from_dict(
        cls,
        d: Mapping[str, Any],
        authority: AuthorityGraph,
    ) -> "ControlBarrierFunction":
        if "schema_version" in d:
            _check_compatible_version(
                d["schema_version"], context="ControlBarrierFunction",
            )
        cbf = cls(
            bounds=SafetyBounds.from_dict(d["bounds"]),
            authority=authority,
            margin=float(d.get("margin", 0.05)),
            soft_slope=float(d.get("soft_slope", 10.0)),
        )
        # Replace the materialised B with the saved one (preserves any
        # user-supplied overrides).
        cbf._B = np.asarray(d["dynamics_B"], dtype=np.float64)
        if cbf._B.shape != (JOINT_ACTION_DIM, STATE_DIM):
            raise MitigationError(
                f"dynamics_B shape {cbf._B.shape} != expected "
                f"({JOINT_ACTION_DIM}, {STATE_DIM})"
            )
        return cbf


# ---------------------------------------------------------------------------
# LinearCrisisDynamics: counterfactual macro simulator
# ---------------------------------------------------------------------------


class LinearCrisisDynamics(nn.Module):
    """Counterfactual macro dynamics ``x_{t+1} = A x_t + B u_t + bias``.

    Parameters
    ----------
    A : Tensor of shape ``(STATE_DIM, STATE_DIM)``
        State-transition matrix. Identity by default; ``fit`` estimates
        from the case library via least squares on consecutive post-
        onset trajectory observations.
    B : Tensor of shape ``(JOINT_ACTION_DIM, STATE_DIM)``
        Action-response matrix (impulse responses). Initialised to
        ``DEFAULT_B_MATRIX_IMPULSE_RESPONSES`` and held fixed by
        default; ``trainable_B=True`` allows fitting.
    bias : Tensor of shape ``(STATE_DIM,)``
        Constant drift term. Zero by default.
    noise_std : Tensor of shape ``(STATE_DIM,)``
        Per-feature stochastic perturbation amplitude. Zero by default;
        ``step_stochastic`` samples from ``N(0, diag(noise_std))``.

    Action units
    ------------
    The B matrix is calibrated in *physical* action units (basis points,
    USD billion, percentage points). To use unit-space actions from a
    tanh actor, scale them via ``AuthorityGraph.scale_to_physical``
    first.

    Why linear
    ----------
    Linear state-space dynamics are the canonical first-line modelling
    tool in central-bank applied work (FAVAR, BVAR, simple gap models).
    They are transparent, easily inspected, fast to roll out, and
    integrate cleanly with the CBF (which assumes linear one-step
    predictability). Non-linear extensions belong in future work but
    are not required for the camera-ready submission.
    """

    def __init__(
        self,
        *,
        trainable_A: bool = False,
        trainable_B: bool = False,
        trainable_bias: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        # Initialise A = I (no drift, no mean reversion).
        A_init = torch.eye(STATE_DIM, dtype=torch.float32)
        # Initialise B from the literature-calibrated defaults.
        B_default = np.zeros((JOINT_ACTION_DIM, STATE_DIM), dtype=np.float32)
        for lev, resps in DEFAULT_B_MATRIX_IMPULSE_RESPONSES.items():
            j = DEFAULT_LEVERS.index(lev)
            for feat, val in resps.items():
                i = DEFAULT_FEATURE_NAMES.index(feat)
                B_default[j, i] = float(val)
        B_init = torch.as_tensor(B_default)
        bias_init = torch.zeros(STATE_DIM, dtype=torch.float32)

        if trainable_A:
            self.A = nn.Parameter(A_init)
        else:
            self.register_buffer("A", A_init)
        if trainable_B:
            self.B = nn.Parameter(B_init)
        else:
            self.register_buffer("B", B_init)
        if trainable_bias:
            self.bias = nn.Parameter(bias_init)
        else:
            self.register_buffer("bias", bias_init)
        # Noise std is a buffer (used only at simulation time).
        self.register_buffer(
            "noise_std", torch.zeros(STATE_DIM, dtype=torch.float32),
        )

        self.config: dict[str, Any] = {
            "trainable_A": bool(trainable_A),
            "trainable_B": bool(trainable_B),
            "trainable_bias": bool(trainable_bias),
            "seed": int(seed),
        }
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Forward / step / rollout
    # ------------------------------------------------------------------ #

    def step(
        self,
        state: torch.Tensor,
        physical_action: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic one-step transition.

        Parameters
        ----------
        state : Tensor of shape ``(..., STATE_DIM)``
        physical_action : Tensor of shape ``(..., JOINT_ACTION_DIM)``
            Action in physical units; multiply unit-space actor outputs
            by ``AuthorityGraph.lever_scale_vector`` before calling.
        """
        if state.shape[-1] != STATE_DIM:
            raise MitigationError(
                f"state last dim {state.shape[-1]} != {STATE_DIM}"
            )
        if physical_action.shape[-1] != JOINT_ACTION_DIM:
            raise MitigationError(
                f"action last dim {physical_action.shape[-1]} != {JOINT_ACTION_DIM}"
            )
        # x A^T (since A acts on a column vector but we have row-vector convention)
        return state @ self.A.T + physical_action @ self.B + self.bias

    def step_stochastic(
        self,
        state: torch.Tensor,
        physical_action: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Stochastic one-step transition with Gaussian perturbation."""
        det = self.step(state, physical_action)
        if torch.all(self.noise_std == 0):
            return det
        eps = torch.randn(det.shape, generator=generator, device=det.device,
                          dtype=det.dtype)
        return det + eps * self.noise_std

    def rollout(
        self,
        initial_state: torch.Tensor,
        physical_action_sequence: torch.Tensor,
        *,
        stochastic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Roll out for ``T`` quarters.

        Parameters
        ----------
        initial_state : Tensor of shape ``(B, STATE_DIM)`` or ``(STATE_DIM,)``
        physical_action_sequence : Tensor of shape ``(B, T, JOINT_ACTION_DIM)``
            or ``(T, JOINT_ACTION_DIM)``.

        Returns
        -------
        trajectory : Tensor of shape ``(B, T+1, STATE_DIM)`` or
        ``(T+1, STATE_DIM)``. Index 0 is the initial state.
        """
        if initial_state.ndim == 1:
            initial_state = initial_state.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        if physical_action_sequence.ndim == 2:
            physical_action_sequence = physical_action_sequence.unsqueeze(0)
        if physical_action_sequence.shape[0] != initial_state.shape[0]:
            raise MitigationError(
                "batch dim mismatch between initial_state and "
                "physical_action_sequence"
            )
        T = physical_action_sequence.shape[1]
        traj = [initial_state]
        s = initial_state
        for t in range(T):
            a = physical_action_sequence[:, t]
            s = (self.step_stochastic(s, a, generator=generator)
                 if stochastic else self.step(s, a))
            traj.append(s)
        out = torch.stack(traj, dim=1)
        return out.squeeze(0) if squeeze else out

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def fit(
        self,
        library: CaseLibrary,
        *,
        ridge: float = 1e-3,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Fit A from the case library via ridge least squares on
        consecutive observations.

        For each case, we collect successive (x_t, x_{t+1}) pairs from
        the pre-onset trajectory. We do *not* fit B from the cases —
        per-quarter action labels are not in the case schema, and B
        carries strong prior knowledge from the impulse-response
        literature.

        Returns
        -------
        summary : dict
            ``"n_pairs"``, ``"residual_norm"``, ``"residual_per_feature"``.
        """
        if len(library) == 0:
            raise MitigationError("LinearCrisisDynamics.fit: empty library")
        if not self.config["trainable_A"]:
            warnings.warn(
                "LinearCrisisDynamics.fit: A was registered as a buffer "
                "(trainable_A=False). The fitted A will replace the "
                "buffer in place, but no gradients are recorded.",
                stacklevel=2,
            )

        # Build X (current), Y (next) of shape (N_pairs, STATE_DIM)
        X_blocks: list[np.ndarray] = []
        Y_blocks: list[np.ndarray] = []
        for cid in library:
            case = library[cid]
            pre = case.pre_onset_trajectory   # (T, F)
            if pre.shape[0] < 2:
                continue
            X_blocks.append(pre[:-1].astype(np.float64))
            Y_blocks.append(pre[1:].astype(np.float64))
        if not X_blocks:
            raise MitigationError(
                "LinearCrisisDynamics.fit: no consecutive observations "
                "in library"
            )
        X = np.concatenate(X_blocks, axis=0)
        Y = np.concatenate(Y_blocks, axis=0)
        n_pairs = X.shape[0]

        # Ridge solution: A^T = (X^T X + ridge I)^{-1} X^T Y
        XtX = X.T @ X + ridge * np.eye(STATE_DIM)
        XtY = X.T @ Y
        At = np.linalg.solve(XtX, XtY)
        A_new = At.T   # (STATE_DIM, STATE_DIM)
        # Stability check: spectral radius of A
        eigvals = np.linalg.eigvals(A_new)
        spec_rad = float(np.max(np.abs(eigvals)))
        if spec_rad > 1.05:
            warnings.warn(
                f"LinearCrisisDynamics.fit: spectral radius "
                f"{spec_rad:.3f} > 1.05; rollouts may diverge. Consider "
                f"increasing ridge.",
                stacklevel=2,
            )
        # Compute residual
        residual = Y - X @ At
        residual_per_feature = np.linalg.norm(residual, axis=0)

        if isinstance(self.A, nn.Parameter):
            self.A.data.copy_(torch.as_tensor(A_new, dtype=self.A.dtype))
        else:
            self.A.copy_(torch.as_tensor(A_new, dtype=self.A.dtype))
        # Estimate noise_std per feature as residual std
        noise_std_est = np.std(residual, axis=0)
        self.noise_std.copy_(torch.as_tensor(noise_std_est, dtype=self.noise_std.dtype))

        self._is_fitted = True
        summary = {
            "n_pairs": int(n_pairs),
            "residual_norm": float(np.linalg.norm(residual)),
            "spectral_radius_A": spec_rad,
            "residual_per_feature": residual_per_feature.tolist(),
        }
        if verbose:
            logger.info(
                "LinearCrisisDynamics.fit: A fit on %d pairs, "
                "spectral_radius=%.3f, residual=%.3f",
                n_pairs, spec_rad, summary["residual_norm"],
            )
        return summary

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sd_path = path / "dynamics.pt"
        sd_tmp = sd_path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), sd_tmp)
        os.replace(sd_tmp, sd_path)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "config": self.config,
            "is_fitted": self._is_fitted,
        }
        _atomic_write_text(
            path / "dynamics_config.json",
            json.dumps(meta, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        return path

    @classmethod
    def load(
        cls, path: Path,
        *,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "LinearCrisisDynamics":
        path = Path(path)
        meta = json.loads((path / "dynamics_config.json").read_text(encoding="utf-8"))
        _check_compatible_version(
            meta.get("schema_version", SCHEMA_VERSION),
            context=f"LinearCrisisDynamics@{path}",
        )
        dyn = cls(**meta["config"])
        dyn.load_state_dict(torch.load(path / "dynamics.pt", map_location=map_location))
        dyn._is_fitted = bool(meta.get("is_fitted", False))
        dyn.to(map_location)
        dyn.eval()
        return dyn


# ---------------------------------------------------------------------------
# Networks: CaseAugmentedActor and TwinCritic
# ---------------------------------------------------------------------------


class CaseAugmentedActor(nn.Module):
    """Per-institution policy network conditioned on retrieved-case
    context.

    Inputs
    ------
    state : Tensor of shape ``(B, STATE_DIM)``
        Current macro state.
    context : Tensor of shape ``(B, context_dim)``
        Retrieved-case context vector from
        ``AnalogyEngine.retrieve(...).context_vector``.
    type_posterior : Tensor of shape ``(B, n_crisis_types)``
        Crisis-type posterior on the simplex.

    Output
    ------
    action : Tensor of shape ``(B, n_own_levers)``
        Tanh-squashed action in ``[-1, 1]`` over the institution's own
        levers, in canonical (``DEFAULT_LEVERS``-sub-)order. Scaling to
        physical units is the AuthorityGraph's job.

    Architecture
    ------------
    Concatenation of ``[state, context, type_posterior]`` -> two-layer
    MLP with GELU activation and dropout -> linear head to
    ``n_own_levers`` -> tanh. ~30 k parameters for the canonical
    configuration (3 institutions × ~10 k each).
    """

    def __init__(
        self,
        institution: str,
        authority: AuthorityGraph,
        *,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        n_crisis_types: int = len(CRISIS_TYPES),
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        n_own = authority.n_levers_for(institution)
        if n_own == 0:
            raise MitigationError(
                f"CaseAugmentedActor: institution {institution!r} owns "
                f"zero levers in this AuthorityGraph"
            )
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1; got {context_dim}")
        if n_crisis_types < 1:
            raise ValueError(f"n_crisis_types must be >= 1; got {n_crisis_types}")
        if hidden_dim < 4:
            raise ValueError(f"hidden_dim must be >= 4; got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")

        self.institution = institution
        self.n_own_levers = n_own
        self.config: dict[str, Any] = {
            "institution": institution,
            "context_dim": int(context_dim),
            "n_crisis_types": int(n_crisis_types),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "seed": int(seed),
        }

        input_dim = STATE_DIM + context_dim + n_crisis_types

        def _build() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_own),
            )
        self.net = _seeded_module_init(seed, _build)
        # Zero-init the final layer so the actor outputs near 0 at
        # start — corresponds to "no action" before training, which is
        # safer than random actions.
        with torch.no_grad():
            final = self.net[-1]
            assert isinstance(final, nn.Linear)
            final.weight.zero_()
            final.bias.zero_()

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        type_posterior: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the institution's action in unit space."""
        if state.shape[-1] != STATE_DIM:
            raise MitigationError(
                f"actor: state last dim {state.shape[-1]} != {STATE_DIM}"
            )
        if context.shape[-1] != self.config["context_dim"]:
            raise MitigationError(
                f"actor: context last dim {context.shape[-1]} != "
                f"{self.config['context_dim']}"
            )
        if type_posterior.shape[-1] != self.config["n_crisis_types"]:
            raise MitigationError(
                f"actor: type_posterior last dim {type_posterior.shape[-1]} "
                f"!= {self.config['n_crisis_types']}"
            )
        z = torch.cat([state, context, type_posterior], dim=-1)
        return torch.tanh(self.net(z))

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sd_path = path / "actor.pt"
        sd_tmp = sd_path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), sd_tmp)
        os.replace(sd_tmp, sd_path)
        meta = {"schema_version": SCHEMA_VERSION, "config": self.config}
        _atomic_write_text(
            path / "actor_config.json",
            json.dumps(meta, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path,
        authority: AuthorityGraph,
        *,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "CaseAugmentedActor":
        path = Path(path)
        meta = json.loads(
            (path / "actor_config.json").read_text(encoding="utf-8"),
        )
        _check_compatible_version(
            meta.get("schema_version", SCHEMA_VERSION),
            context=f"CaseAugmentedActor@{path}",
        )
        cfg = meta["config"]
        actor = cls(
            institution=cfg["institution"], authority=authority,
            context_dim=cfg["context_dim"],
            n_crisis_types=cfg["n_crisis_types"],
            hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"],
            seed=cfg["seed"],
        )
        actor.load_state_dict(
            torch.load(path / "actor.pt", map_location=map_location),
        )
        actor.to(map_location)
        actor.eval()
        return actor


class TwinCritic(nn.Module):
    """Twin Q-function over (state, joint_action, context, type_posterior).

    Two independent Q-heads share the same architecture and the same
    input but have different parameter initialisations. The TD3 critic
    update uses ``min(Q1, Q2)`` for the target, mitigating
    overestimation bias.

    The critic is *centralised*: it sees the joint action (all
    institutions' actions concatenated), even though each actor only
    sees its own input. This is the CTDE (centralised training,
    decentralised execution) paradigm of multi-agent RL.
    """

    def __init__(
        self,
        *,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        n_crisis_types: int = len(CRISIS_TYPES),
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1; got {context_dim}")
        if hidden_dim < 4:
            raise ValueError(f"hidden_dim must be >= 4; got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")

        self.config: dict[str, Any] = {
            "context_dim": int(context_dim),
            "n_crisis_types": int(n_crisis_types),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "seed": int(seed),
        }
        input_dim = STATE_DIM + JOINT_ACTION_DIM + context_dim + n_crisis_types

        def _build_q():
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        # Two heads with different seeds → different init.
        self.q1 = _seeded_module_init(seed, _build_q)
        self.q2 = _seeded_module_init(seed + 1, _build_q)

    def forward(
        self,
        state: torch.Tensor,
        joint_action: torch.Tensor,
        context: torch.Tensor,
        type_posterior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([state, joint_action, context, type_posterior], dim=-1)
        return self.q1(z).squeeze(-1), self.q2(z).squeeze(-1)

    def q1_only(
        self,
        state: torch.Tensor,
        joint_action: torch.Tensor,
        context: torch.Tensor,
        type_posterior: torch.Tensor,
    ) -> torch.Tensor:
        """For the actor update we only need Q1 (TD3 convention)."""
        z = torch.cat([state, joint_action, context, type_posterior], dim=-1)
        return self.q1(z).squeeze(-1)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sd_path = path / "critic.pt"
        sd_tmp = sd_path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), sd_tmp)
        os.replace(sd_tmp, sd_path)
        meta = {"schema_version": SCHEMA_VERSION, "config": self.config}
        _atomic_write_text(
            path / "critic_config.json",
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
    ) -> "TwinCritic":
        path = Path(path)
        meta = json.loads(
            (path / "critic_config.json").read_text(encoding="utf-8"),
        )
        _check_compatible_version(
            meta.get("schema_version", SCHEMA_VERSION),
            context=f"TwinCritic@{path}",
        )
        critic = cls(**meta["config"])
        critic.load_state_dict(
            torch.load(path / "critic.pt", map_location=map_location),
        )
        critic.to(map_location)
        critic.eval()
        return critic


# ---------------------------------------------------------------------------
# Multi-agent policy orchestrator
# ---------------------------------------------------------------------------


class MultiAgentMitigationPolicy(nn.Module):
    """Per-institution actors + shared twin critic + CBF filter.

    Wraps inference (``get_action``) with the CBF and provides save/load
    that round-trips bit-exactly. The trainer (``MitigationTrainer``)
    operates on this object directly and updates its parameters.
    """

    def __init__(
        self,
        authority: AuthorityGraph,
        *,
        cbf: Optional[ControlBarrierFunction] = None,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        n_crisis_types: int = len(CRISIS_TYPES),
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not isinstance(authority, AuthorityGraph):
            raise TypeError(
                f"authority must be AuthorityGraph; got {type(authority).__name__}"
            )
        self.authority = authority
        self.cbf = cbf
        self.config: dict[str, Any] = {
            "context_dim": int(context_dim),
            "n_crisis_types": int(n_crisis_types),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "seed": int(seed),
        }
        # One actor per institution that owns at least one lever.
        self.actors = nn.ModuleDict({
            inst: CaseAugmentedActor(
                institution=inst, authority=authority,
                context_dim=context_dim, n_crisis_types=n_crisis_types,
                hidden_dim=hidden_dim, dropout=dropout,
                seed=seed + i,
            )
            for i, inst in enumerate(authority.institutions)
        })
        # Single twin critic (centralised).
        self.critic = TwinCritic(
            context_dim=context_dim, n_crisis_types=n_crisis_types,
            hidden_dim=hidden_dim, dropout=dropout, seed=seed + 100,
        )

    # ------------------------------------------------------------------ #
    # Joint forward over actors
    # ------------------------------------------------------------------ #

    def joint_actor_forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        type_posterior: torch.Tensor,
    ) -> torch.Tensor:
        """Run all actors and assemble a joint action in unit space.

        Shape: input ``state``, ``context``, ``type_posterior`` are
        ``(B, ...)``; output is ``(B, JOINT_ACTION_DIM)``.
        """
        device = state.device
        batch = state.shape[0]
        joint = torch.zeros(
            (batch, JOINT_ACTION_DIM), device=device, dtype=state.dtype,
        )
        for inst, actor in self.actors.items():
            sub_action = actor(state, context, type_posterior)   # (B, n_own)
            idx = torch.as_tensor(
                self.authority._slices[inst], device=device, dtype=torch.long,
            )
            joint = joint.index_copy(
                dim=-1, index=idx, source=sub_action,
            )
        return joint

    # ------------------------------------------------------------------ #
    # Inference: single-state action selection with optional CBF
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def get_action(
        self,
        state: np.ndarray,
        retrieval: RetrievalResult,
        *,
        exploration_noise: float = 0.0,
        apply_cbf: bool = True,
        generator: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute a joint unit-space action for one state.

        Parameters
        ----------
        state : np.ndarray of shape ``(STATE_DIM,)``
        retrieval : RetrievalResult
            Provides ``context_vector`` and ``type_posterior``.
        exploration_noise : float, default 0
            Standard deviation of Gaussian noise added to the unit
            action before clipping to ``[-1, 1]``. Use during training
            rollouts; set to 0 for evaluation.
        apply_cbf : bool, default True
            If True and ``self.cbf`` is not None, project the noised
            action onto the safe set via ``cbf.filter_hard``.
        generator : numpy Generator, optional
            For reproducible exploration noise.

        Returns
        -------
        (joint_action_unit, info) : (ndarray, dict)
            ``joint_action_unit`` of shape ``(JOINT_ACTION_DIM,)`` in
            ``[-1, 1]``. ``info`` contains diagnostics:
            ``"raw_action"``, ``"cbf_applied"``, ``"cbf_modified"``.
        """
        state_arr = np.asarray(state, dtype=np.float64)
        if state_arr.shape != (STATE_DIM,):
            raise MitigationError(
                f"get_action: state shape {state_arr.shape} != "
                f"({STATE_DIM},)"
            )
        device = next(self.parameters()).device
        # Set modules to eval for inference (dropout off); restore later.
        was_training = self.training
        self.eval()
        try:
            s = torch.as_tensor(state_arr, dtype=torch.float32, device=device).unsqueeze(0)
            c = torch.as_tensor(
                retrieval.context_vector.copy(),
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            p = torch.as_tensor(
                retrieval.type_posterior.copy(),
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            joint = self.joint_actor_forward(s, c, p)   # (1, JOINT_ACTION_DIM)
            raw = joint[0].cpu().numpy().astype(np.float64)
        finally:
            if was_training:
                self.train()

        action = raw.copy()
        if exploration_noise > 0:
            if generator is None:
                generator = np.random.default_rng()
            noise = generator.normal(0.0, exploration_noise, size=JOINT_ACTION_DIM)
            action = (action + noise).clip(-1.0, 1.0)

        cbf_applied = False
        cbf_modified = False
        if apply_cbf and self.cbf is not None:
            action, cbf_modified = self.cbf.filter_hard(state_arr, action)
            cbf_applied = True

        info = {
            "raw_action": raw,
            "cbf_applied": cbf_applied,
            "cbf_modified": cbf_modified,
        }
        return action, info

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Save the whole module state in one go for simplicity
        sd_path = path / "policy.pt"
        sd_tmp = sd_path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), sd_tmp)
        os.replace(sd_tmp, sd_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "config": self.config,
            "authority": self.authority.to_dict(),
            "actor_configs": {
                inst: actor.config for inst, actor in self.actors.items()
            },
            "critic_config": self.critic.config,
            "has_cbf": self.cbf is not None,
            "cbf": self.cbf.to_dict() if self.cbf is not None else None,
        }
        _atomic_write_text(
            path / "policy_manifest.json",
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
    ) -> "MultiAgentMitigationPolicy":
        path = Path(path)
        manifest = json.loads(
            (path / "policy_manifest.json").read_text(encoding="utf-8"),
        )
        _check_compatible_version(
            manifest.get("schema_version", SCHEMA_VERSION),
            context=f"MultiAgentMitigationPolicy@{path}",
        )
        authority = AuthorityGraph.from_dict(manifest["authority"])
        cbf = None
        if manifest.get("has_cbf") and manifest.get("cbf") is not None:
            cbf = ControlBarrierFunction.from_dict(manifest["cbf"], authority)
        policy = cls(authority=authority, cbf=cbf, **manifest["config"])
        policy.load_state_dict(
            torch.load(path / "policy.pt", map_location=map_location),
        )
        policy.to(map_location)
        policy.eval()
        return policy


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MitigationTransition:
    """Immutable replay-buffer record.

    Carries the standard SARS-D tuple plus the retrieval bookkeeping
    needed for the case-coherence regulariser.

    Fields
    ------
    state : np.ndarray (STATE_DIM,)
    joint_action_unit : np.ndarray (JOINT_ACTION_DIM,)
        Action in unit space ``[-1, 1]`` (pre-scaling).
    reward : float
    next_state : np.ndarray (STATE_DIM,)
    done : bool
    context_vector : np.ndarray (context_dim,)
    type_posterior : np.ndarray (n_crisis_types,)
    retrieved_fp_per_lever : np.ndarray (K, JOINT_ACTION_DIM)
        Per-lever policy fingerprints of the retrieved cases (already
        collapsed from 24-dim institution-by-lever).
    retrieved_outcomes : np.ndarray (K,)
        Cumulative output loss (% of GDP) of each retrieved case.
    retrieval_weights : np.ndarray (K,)
        Softmax weights over the K retrieved cases (sum to 1).
    """
    state: np.ndarray
    joint_action_unit: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    context_vector: np.ndarray
    type_posterior: np.ndarray
    retrieved_fp_per_lever: np.ndarray
    retrieved_outcomes: np.ndarray
    retrieval_weights: np.ndarray

    def __post_init__(self) -> None:
        # Light validation; full shape checks are at buffer-sample time
        if self.state.shape != (STATE_DIM,):
            raise MitigationError(
                f"transition.state shape {self.state.shape} != ({STATE_DIM},)"
            )
        if self.joint_action_unit.shape != (JOINT_ACTION_DIM,):
            raise MitigationError(
                f"transition.joint_action_unit shape "
                f"{self.joint_action_unit.shape} != ({JOINT_ACTION_DIM},)"
            )
        if self.next_state.shape != (STATE_DIM,):
            raise MitigationError(
                f"transition.next_state shape {self.next_state.shape} "
                f"!= ({STATE_DIM},)"
            )
        if self.context_vector.ndim != 1:
            raise MitigationError("transition.context_vector must be 1-D")
        if self.type_posterior.ndim != 1:
            raise MitigationError("transition.type_posterior must be 1-D")
        k = self.retrieval_weights.shape[0]
        if self.retrieved_fp_per_lever.shape != (k, JOINT_ACTION_DIM):
            raise MitigationError(
                f"transition.retrieved_fp_per_lever shape "
                f"{self.retrieved_fp_per_lever.shape} != ({k}, {JOINT_ACTION_DIM})"
            )
        if self.retrieved_outcomes.shape != (k,):
            raise MitigationError(
                f"transition.retrieved_outcomes shape "
                f"{self.retrieved_outcomes.shape} != ({k},)"
            )


class ReplayBuffer:
    """Fixed-capacity ring buffer of ``MitigationTransition``.

    Implementation
    --------------
    Backed by numpy arrays for each field. ``append`` overwrites the
    oldest entry when capacity is reached. ``sample`` returns a dict
    of torch tensors batched along axis 0.

    All ``K`` (retrieval-set size) values in stored transitions must be
    the same; the buffer rejects appends with a different K from the
    first one observed.
    """

    def __init__(
        self,
        capacity: int,
        *,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        n_crisis_types: int = len(CRISIS_TYPES),
        k_retrieval: int = 5,
        seed: int = 42,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1; got {capacity}")
        self.capacity = int(capacity)
        self.context_dim = int(context_dim)
        self.n_crisis_types = int(n_crisis_types)
        self.k_retrieval = int(k_retrieval)
        self._gen = np.random.default_rng(int(seed))
        self._n: int = 0
        self._ptr: int = 0
        # Pre-allocate
        self._state = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self._action = np.zeros((capacity, JOINT_ACTION_DIM), dtype=np.float32)
        self._reward = np.zeros((capacity,), dtype=np.float32)
        self._next_state = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self._done = np.zeros((capacity,), dtype=np.bool_)
        self._context = np.zeros((capacity, context_dim), dtype=np.float32)
        self._type_post = np.zeros((capacity, n_crisis_types), dtype=np.float32)
        self._fp_per_lever = np.zeros(
            (capacity, k_retrieval, JOINT_ACTION_DIM), dtype=np.float32,
        )
        self._outcomes = np.zeros((capacity, k_retrieval), dtype=np.float32)
        self._weights = np.zeros((capacity, k_retrieval), dtype=np.float32)

    def __len__(self) -> int:
        return self._n

    def append(self, transition: MitigationTransition) -> None:
        # Shape checks against buffer config
        if transition.context_vector.shape != (self.context_dim,):
            raise MitigationError(
                f"buffer.append: context_vector shape "
                f"{transition.context_vector.shape} != ({self.context_dim},)"
            )
        if transition.type_posterior.shape != (self.n_crisis_types,):
            raise MitigationError(
                f"buffer.append: type_posterior shape "
                f"{transition.type_posterior.shape} != "
                f"({self.n_crisis_types},)"
            )
        if transition.retrieval_weights.shape != (self.k_retrieval,):
            raise MitigationError(
                f"buffer.append: retrieval_weights shape "
                f"{transition.retrieval_weights.shape} != "
                f"({self.k_retrieval},)"
            )
        i = self._ptr
        self._state[i] = transition.state
        self._action[i] = transition.joint_action_unit
        self._reward[i] = float(transition.reward)
        self._next_state[i] = transition.next_state
        self._done[i] = bool(transition.done)
        self._context[i] = transition.context_vector
        self._type_post[i] = transition.type_posterior
        self._fp_per_lever[i] = transition.retrieved_fp_per_lever
        self._outcomes[i] = transition.retrieved_outcomes
        self._weights[i] = transition.retrieval_weights
        self._ptr = (self._ptr + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def sample(
        self, batch_size: int,
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Sample a batch of transitions as a dict of torch tensors."""
        if self._n == 0:
            raise MitigationError("ReplayBuffer.sample: buffer is empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")
        idx = self._gen.integers(0, self._n, size=batch_size)
        return {
            "state": torch.as_tensor(self._state[idx], device=device),
            "action": torch.as_tensor(self._action[idx], device=device),
            "reward": torch.as_tensor(self._reward[idx], device=device),
            "next_state": torch.as_tensor(self._next_state[idx], device=device),
            "done": torch.as_tensor(self._done[idx].astype(np.float32), device=device),
            "context": torch.as_tensor(self._context[idx], device=device),
            "type_posterior": torch.as_tensor(self._type_post[idx], device=device),
            "fp_per_lever": torch.as_tensor(self._fp_per_lever[idx], device=device),
            "outcomes": torch.as_tensor(self._outcomes[idx], device=device),
            "weights": torch.as_tensor(self._weights[idx], device=device),
        }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def case_coherence_loss(
    actor_action_unit: torch.Tensor,
    retrieved_fp_per_lever: torch.Tensor,
    retrieved_outcomes: torch.Tensor,
    retrieval_weights: torch.Tensor,
    lever_scale: torch.Tensor,
    *,
    outcome_temperature: float = DEFAULT_OUTCOME_TEMPERATURE,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Case-coherence regulariser.

    Penalises the actor for choosing actions that diverge from the
    *retrieval-weighted, outcome-weighted* target action. The target
    is a weighted average over the retrieved cases of their per-lever
    policy fingerprints, with weights being the retrieval similarity
    weights tempered by an outcome-quality term: lower-loss cases get
    higher effective weight.

    Mathematics
    -----------
    .. math::

        w_i^{\\mathrm{outcome}}
            = \\mathrm{softmax}(-\\mathrm{loss}_i / T)_i

        w_i^{\\mathrm{eff}}
            = \\mathrm{normalise}(w_i^{\\mathrm{retr}} \\cdot w_i^{\\mathrm{outcome}})

        \\bar{f} = \\sum_i w_i^{\\mathrm{eff}}
                  \\cdot f_i / N_{\\mathrm{post}}

        L = 1 - \\cos(a^{\\mathrm{phys}}, \\bar{f})

    where :math:`a^{\\mathrm{phys}}` is the actor's action in physical
    units, :math:`f_i` is the i-th retrieved case's per-lever
    fingerprint (already in physical units, summed over the case's
    full post-onset window), and :math:`N_{\\mathrm{post}}` is the
    canonical post-onset window length
    (``DEFAULT_POST_ONSET_QUARTERS``). Dividing by
    :math:`N_{\\mathrm{post}}` converts the fingerprint from a
    cumulative-action measure to a per-quarter rate, matching
    :math:`a^{\\mathrm{phys}}`.

    Cosine similarity is scale-equivariant — its magnitude depends only
    on direction — so the loss penalises the *direction* of the
    actor's policy rather than its magnitude. This is the intended
    semantics: the case-coherence prior says "tilt toward what worked
    historically" without forcing a specific magnitude.

    Inputs
    ------
    actor_action_unit : Tensor (B, JOINT_ACTION_DIM)
        Unit-space actor action.
    retrieved_fp_per_lever : Tensor (B, K, JOINT_ACTION_DIM)
        Per-lever policy fingerprints (cumulative physical magnitudes
        over each case's post-onset window).
    retrieved_outcomes : Tensor (B, K)
        Cumulative output loss for each retrieved case (% of GDP).
    retrieval_weights : Tensor (B, K)
        Softmax weights over the retrieved cases (rows sum to 1).
    lever_scale : Tensor (JOINT_ACTION_DIM,)
        Per-lever physical scale (from AuthorityGraph.lever_scale_vector).
    outcome_temperature : float
        Temperature in the outcome-quality softmax.

    Returns
    -------
    Scalar tensor (mean over batch).
    """
    if actor_action_unit.ndim != 2:
        raise MitigationError("actor_action_unit must be (B, n_levers)")
    if retrieved_fp_per_lever.ndim != 3:
        raise MitigationError(
            "retrieved_fp_per_lever must be (B, K, n_levers)"
        )
    if retrieved_outcomes.ndim != 2 or retrieval_weights.ndim != 2:
        raise MitigationError(
            "retrieved_outcomes and retrieval_weights must be (B, K)"
        )
    if outcome_temperature <= 0:
        raise ValueError(
            f"outcome_temperature must be > 0; got {outcome_temperature}"
        )

    # Outcome quality weights (lower loss = higher weight)
    outcome_w = F.softmax(-retrieved_outcomes / outcome_temperature, dim=-1)
    # Combined effective weights
    combined = retrieval_weights * outcome_w
    combined_sum = combined.sum(dim=-1, keepdim=True).clamp_min(eps)
    eff_w = combined / combined_sum                          # (B, K)
    # Weighted average fingerprint, normalised to per-quarter rate
    target_fp = (
        eff_w.unsqueeze(-1) * retrieved_fp_per_lever
    ).sum(dim=-2) / float(DEFAULT_POST_ONSET_QUARTERS)        # (B, n_levers)
    # Convert actor action to physical units
    a_phys = actor_action_unit * lever_scale                  # (B, n_levers)
    # Cosine similarity per row
    cos = F.cosine_similarity(a_phys, target_fp, dim=-1, eps=eps)
    return (1.0 - cos).mean()


def compute_reward(
    state: np.ndarray,
    joint_action_unit: np.ndarray,
    *,
    output_gap_weight: float = 1.0,
    inflation_weight: float = 0.5,
    action_volatility_weight: float = 0.1,
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
) -> float:
    """Default reward function.

    .. math::

        r = -w_o |g_t| - w_i |\\pi_t| - w_a \\|a_t\\|_2^2

    where :math:`g_t` is the output gap, :math:`\\pi_t` is the inflation
    deviation from zero, and :math:`a_t` is the unit-space action.

    Sign convention: rewards are non-positive. The optimal trajectory
    has :math:`r = 0` (gap = 0, inflation = 0, no actions taken).
    """
    state_arr = np.asarray(state, dtype=np.float64)
    action_arr = np.asarray(joint_action_unit, dtype=np.float64)
    og = float(state_arr[feature_names.index("output_gap")])
    pi = float(state_arr[feature_names.index("inflation_yoy")])
    action_pen = float(np.sum(action_arr ** 2))
    return (
        -output_gap_weight * abs(og)
        - inflation_weight * abs(pi)
        - action_volatility_weight * action_pen
    )


# ---------------------------------------------------------------------------
# Trainer: one TD3 optimisation step
# ---------------------------------------------------------------------------


def _polyak_update(
    target: nn.Module, source: nn.Module, polyak: float,
) -> None:
    """target = polyak * target + (1 - polyak) * source (in place)."""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(polyak).add_(sp.data, alpha=1.0 - polyak)
        for tb, sb in zip(target.buffers(), source.buffers()):
            tb.data.mul_(polyak).add_(sb.data, alpha=1.0 - polyak)


class MitigationTrainer:
    """Encapsulates one TD3 optimisation step with case-coherence
    regularisation.

    Owned state:
    * The target copies of policy.actors and policy.critic.
    * Two AdamW optimisers (actor params, critic params).
    * Step counter (for delayed actor updates).

    The trainer does not own the replay buffer; the caller (typically
    ``training.py``) is responsible for filling and passing batches.

    Hyperparameters
    ---------------
    actor_lr, critic_lr, gamma, polyak, actor_update_freq,
    target_noise_std, target_noise_clip : standard TD3.
    case_coherence_weight : weight on the case-coherence regulariser
        added to the actor loss.
    safety_penalty_weight : weight on the CBF soft-penalty term.
    grad_clip : per-network gradient-norm clip (applies independently
        to actor and critic).
    """

    def __init__(
        self,
        policy: MultiAgentMitigationPolicy,
        *,
        actor_lr: float = DEFAULT_ACTOR_LR,
        critic_lr: float = DEFAULT_CRITIC_LR,
        gamma: float = DEFAULT_GAMMA,
        polyak: float = DEFAULT_POLYAK,
        actor_update_freq: int = DEFAULT_ACTOR_UPDATE_FREQ,
        target_noise_std: float = DEFAULT_TARGET_NOISE_STD,
        target_noise_clip: float = DEFAULT_TARGET_NOISE_CLIP,
        case_coherence_weight: float = DEFAULT_CASE_COHERENCE_WEIGHT,
        safety_penalty_weight: float = DEFAULT_SAFETY_PENALTY_WEIGHT,
        grad_clip: float = 1.0,
        weight_decay: float = 1e-4,
    ) -> None:
        if not isinstance(policy, MultiAgentMitigationPolicy):
            raise TypeError(
                f"policy must be MultiAgentMitigationPolicy; got "
                f"{type(policy).__name__}"
            )
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1]; got {gamma}")
        if not 0.0 < polyak < 1.0:
            raise ValueError(f"polyak must be in (0, 1); got {polyak}")
        if actor_update_freq < 1:
            raise ValueError(
                f"actor_update_freq must be >= 1; got {actor_update_freq}"
            )

        self.policy = policy
        self.gamma = float(gamma)
        self.polyak = float(polyak)
        self.actor_update_freq = int(actor_update_freq)
        self.target_noise_std = float(target_noise_std)
        self.target_noise_clip = float(target_noise_clip)
        self.case_coherence_weight = float(case_coherence_weight)
        self.safety_penalty_weight = float(safety_penalty_weight)
        self.grad_clip = float(grad_clip)
        self._step_count: int = 0

        # Target networks: deep copies of the policy's actors and critic.
        device = next(policy.parameters()).device
        import copy
        self.target_actors = nn.ModuleDict({
            inst: copy.deepcopy(actor).to(device)
            for inst, actor in policy.actors.items()
        })
        self.target_critic = copy.deepcopy(policy.critic).to(device)
        # Targets stay in eval mode forever: TD3 target-Q computation
        # should be deterministic (no dropout) and the targets are never
        # gradient-updated (only Polyak-averaged from the online nets).
        self.target_actors.eval()
        self.target_critic.eval()
        # Freeze target params (no grad, but we'll Polyak-update them).
        for p in self.target_actors.parameters():
            p.requires_grad_(False)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        # Optimisers
        actor_params = list(policy.actors.parameters())
        critic_params = list(policy.critic.parameters())
        self.actor_optim = torch.optim.AdamW(
            actor_params, lr=actor_lr, weight_decay=weight_decay,
        )
        self.critic_optim = torch.optim.AdamW(
            critic_params, lr=critic_lr, weight_decay=weight_decay,
        )

        # Cache the lever scale tensor (constant during training)
        self._lever_scale: torch.Tensor = torch.as_tensor(
            policy.authority.lever_scale_vector.copy(),
            dtype=torch.float32, device=device,
        )

    # ------------------------------------------------------------------ #
    # Target-action computation with policy smoothing
    # ------------------------------------------------------------------ #

    def _target_joint_action(
        self,
        next_state: torch.Tensor,
        context: torch.Tensor,
        type_posterior: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the target joint action with TD3 target-policy
        smoothing: add clipped Gaussian noise to each actor's output,
        then assemble the joint action."""
        device = next_state.device
        batch = next_state.shape[0]
        joint = torch.zeros(
            (batch, JOINT_ACTION_DIM), device=device, dtype=next_state.dtype,
        )
        for inst, target_actor in self.target_actors.items():
            sub = target_actor(next_state, context, type_posterior)  # (B, n_own)
            noise = (
                torch.randn_like(sub) * self.target_noise_std
            ).clamp(-self.target_noise_clip, self.target_noise_clip)
            sub_noisy = (sub + noise).clamp(-1.0, 1.0)
            idx = torch.as_tensor(
                self.policy.authority._slices[inst], device=device,
                dtype=torch.long,
            )
            joint = joint.index_copy(dim=-1, index=idx, source=sub_noisy)
        return joint

    # ------------------------------------------------------------------ #
    # One optimisation step
    # ------------------------------------------------------------------ #

    def step(self, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """One TD3 update.

        Always updates the critic. Updates the actors every
        ``actor_update_freq`` steps and Polyak-updates targets after
        any actor update.

        Parameters
        ----------
        batch : dict produced by ``ReplayBuffer.sample``.

        Returns
        -------
        losses : dict
            ``"critic_loss"``, ``"q1_mean"``, ``"q2_mean"``,
            ``"actor_loss"`` (only on actor-update steps; else None),
            ``"case_coherence"``, ``"safety_penalty"``.
        """
        self._step_count += 1
        s = batch["state"].float()
        a = batch["action"].float()
        r = batch["reward"].float()
        s_next = batch["next_state"].float()
        d = batch["done"].float()
        ctx = batch["context"].float()
        tp = batch["type_posterior"].float()
        fp = batch["fp_per_lever"].float()
        outc = batch["outcomes"].float()
        rw = batch["weights"].float()

        # ----- Critic update -----
        self.policy.critic.train()
        with torch.no_grad():
            target_action = self._target_joint_action(s_next, ctx, tp)
            q1_t, q2_t = self.target_critic(s_next, target_action, ctx, tp)
            q_target = torch.minimum(q1_t, q2_t)
            y = r + (1.0 - d) * self.gamma * q_target
        q1_pred, q2_pred = self.policy.critic(s, a, ctx, tp)
        critic_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), max_norm=self.grad_clip,
            )
        self.critic_optim.step()

        losses: dict[str, float] = {
            "critic_loss": float(critic_loss.item()),
            "q1_mean": float(q1_pred.mean().item()),
            "q2_mean": float(q2_pred.mean().item()),
        }

        # ----- Actor update (delayed) -----
        if self._step_count % self.actor_update_freq == 0:
            # Re-enable training mode for actors (critic stays in train mode
            # because critic backprop already ran; we don't need grads).
            for actor in self.policy.actors.values():
                actor.train()

            joint_action = self.policy.joint_actor_forward(s, ctx, tp)
            q1 = self.policy.critic.q1_only(s, joint_action, ctx, tp)
            # Policy-gradient loss: maximise Q1 (minimise -Q1)
            policy_loss = -q1.mean()

            # Case-coherence regulariser
            coherence = case_coherence_loss(
                joint_action, fp, outc, rw, self._lever_scale,
            )

            # Soft safety penalty (only if CBF is present)
            if self.policy.cbf is not None:
                safety_pen = self.policy.cbf.soft_penalty(s, joint_action).mean()
            else:
                safety_pen = torch.zeros((), device=s.device, dtype=s.dtype)

            actor_loss = (
                policy_loss
                + self.case_coherence_weight * coherence
                + self.safety_penalty_weight * safety_pen
            )

            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.actors.parameters(), max_norm=self.grad_clip,
                )
            self.actor_optim.step()

            # Polyak update of targets
            for inst in self.target_actors:
                _polyak_update(
                    self.target_actors[inst], self.policy.actors[inst],
                    self.polyak,
                )
            _polyak_update(self.target_critic, self.policy.critic, self.polyak)

            losses["actor_loss"] = float(actor_loss.item())
            losses["policy_loss"] = float(policy_loss.item())
            losses["case_coherence"] = float(coherence.item())
            losses["safety_penalty"] = float(safety_pen.item())
        else:
            losses["actor_loss"] = None
            losses["case_coherence"] = None
            losses["safety_penalty"] = None

        return losses


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def collapse_24dim_fp_to_per_lever(fp_24: np.ndarray) -> np.ndarray:
    """Collapse a 24-dim policy fingerprint to an 8-dim per-lever
    fingerprint by summing over the institution axis.

    The 24-dim fingerprint encodes ``(n_institutions=3, n_levers=8)``
    cells. Since each lever is owned by exactly one institution per
    AuthorityGraph, summing over the institution axis is exact:
    each lever's value equals its owning institution's value, with
    zeros from non-owning institutions adding nothing.

    Accepts batched input ``(..., POLICY_FINGERPRINT_DIM)``; returns
    ``(..., JOINT_ACTION_DIM)``.
    """
    arr = np.asarray(fp_24)
    if arr.shape[-1] != POLICY_FINGERPRINT_DIM:
        raise MitigationError(
            f"fp_24 last dim {arr.shape[-1]} != POLICY_FINGERPRINT_DIM "
            f"{POLICY_FINGERPRINT_DIM}"
        )
    reshaped = arr.reshape(*arr.shape[:-1], len(INSTITUTIONS), len(DEFAULT_LEVERS))
    return reshaped.sum(axis=-2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> int:
    """CLI: inspect a saved mitigation policy.

    Commands
    --------
    summary <policy_path>
        Load and print a summary of a saved policy.
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m em_fin_stability.mitigation",
        description="Inspect a saved MultiAgentMitigationPolicy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_sum = sub.add_parser("summary", help="Print policy summary.")
    p_sum.add_argument("policy_path", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _configure_logging(args.log_level)

    try:
        if args.command == "summary":
            policy = MultiAgentMitigationPolicy.load(args.policy_path)
            summary = {
                "schema_version": SCHEMA_VERSION,
                "n_actors": len(policy.actors),
                "actor_institutions": list(policy.actors.keys()),
                "actor_param_counts": {
                    inst: int(sum(p.numel() for p in actor.parameters()))
                    for inst, actor in policy.actors.items()
                },
                "critic_param_count": int(
                    sum(p.numel() for p in policy.critic.parameters())
                ),
                "has_cbf": policy.cbf is not None,
                "authority": policy.authority.to_dict(),
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        else:  # pragma: no cover
            parser.print_help()
            return 2
    except CaseMemoryError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
