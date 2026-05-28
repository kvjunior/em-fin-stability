"""
case_memory.py
==============

Crisis Case Memory: a structured, versioned, auditable knowledge base of
historical emerging-market financial crises (novelty #1 of the paper).

This module is the centrepiece of the *Knowledge-Aware* framing. The
existing machine-learning literature on financial-crisis prediction treats
each crisis episode as one row in a panel — a positive label to be
classified, not a *case* to be reasoned about. Central-bank desk officers
do the opposite: when an emerging-market crisis is brewing, the first
question on the desk is *"which previous episode does this most resemble,
and what did and did not work then?"*. This module formalises that
analogical-reasoning substrate as a first-class, persistable, queryable
knowledge artefact whose schema is part of the contribution.

Architecture
------------
::

    PolicyAction              -> one immutable (date, institution, lever,
                                 magnitude) record in a case's policy
                                 timeline. Sorted chronologically inside
                                 each CrisisCase.
    AuthoritySnapshot         -> the institution -> levers mapping in
                                 force at the time of the case (because
                                 authority graphs change: Türkiye 2018
                                 had a different mandate structure than
                                 Türkiye 2001).
    CrisisCase                -> the immutable historical fact: country,
                                 onset, crisis type, the 20-quarter
                                 pre-onset macro trajectory, the
                                 12-quarter post-onset realised outcome,
                                 the policy timeline, the authority
                                 snapshot, provenance, and a data-quality
                                 flag.
    CaseLibrary               -> a directory-backed collection of cases
                                 with content-hash checksums, atomic
                                 save/load, query interface, train/test
                                 splits, and schema-version migration.
    CaseSignatureEncoder      -> a small neural network that maps a raw
                                 (T_pre, N_FEATURES) trajectory to a
                                 unit-norm dense signature vector in
                                 R^signature_dim. Trained contrastively
                                 (SupCon) against crisis-type labels so
                                 that signatures are crisis-type
                                 discriminative.
    build_signature_matrix    -> module-level convenience; produces the
                                 (case_ids, signature_matrix) pair
                                 consumed by ``analogy_engine.py`` at
                                 retrieval time.

The case schema is the contribution
-----------------------------------
The novelty here is not the encoder (small CNN+attention; nothing
remarkable on its own) but the *schema* and the commitment to treat the
library as a knowledge artefact that survives encoder versions, model
versions, and even framework versions. Every CrisisCase records:

* Provenance for the binary crisis label (e.g. Laeven-Valencia 2020
  Table 1 row 47), the policy timeline (e.g. IMF MONA program records),
  and the output-loss measurement (e.g. WEO October-2024 vintage).
* A data-quality score in [0, 1] so downstream retrieval can
  down-weight noisy cases.
* The schema version, so future migrations are explicit.
* The created-at timestamp, so the library has temporal coherence.

A reviewer or a central-bank user can inspect any single ``.case.json``
file by eye and verify it against primary sources. This auditability is
the property that makes the case library a *knowledge base* rather than
a *dataset*. We argue throughout the paper that this distinction is the
core difference between an explainable decision-support system and a
black-box classifier.

Reproducibility commitments
---------------------------
* Every case is serialised as canonical JSON: keys sorted, no platform-
  dependent whitespace, Python-native floats (which have stable
  cross-platform repr since Python 3).
* Every case carries a SHA-256 content checksum; the library carries a
  second checksum over the sorted (case_id, case_checksum) pairs. Both
  are recorded in ``library_manifest.json``.
* The encoder records its full constructor configuration in
  ``encoder_config.json`` so that ``CaseSignatureEncoder.load`` re-
  instantiates exactly the architecture that was trained.
* Disk writes are atomic (``tmp`` + ``os.replace``) so an interrupted
  save never corrupts a library.
* All RNG-dependent computations (encoder training, train/test split)
  consume an explicit seed; no implicit use of ``np.random.seed`` or
  ``torch.manual_seed`` at module scope.

References (APA-7)
------------------
Abadie, A. (2021). Using synthetic controls: Feasibility, data
    requirements, and methodological aspects. Journal of Economic
    Literature, 59(2), 391-425.

Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P.,
    Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised
    contrastive learning. In Advances in Neural Information Processing
    Systems 33 (pp. 18661-18673).

Laeven, L., & Valencia, F. (2020). Systemic banking crises database II.
    IMF Economic Review, 68(2), 307-361.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V.,
    Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel,
    S., & Kiela, D. (2020). Retrieval-augmented generation for
    knowledge-intensive NLP tasks. In Advances in Neural Information
    Processing Systems 33 (pp. 9459-9474).

Reinhart, C. M., & Rogoff, K. S. (2009). This time is different: Eight
    centuries of financial folly. Princeton University Press.

Version
-------
1.0.0  Camera-ready KBS submission.

       Initial schema. Future revisions will bump SCHEMA_VERSION and
       must implement an explicit migration path documented in
       ``_migrate_case_dict``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "CRISIS_TYPES",
    "INSTITUTIONS",
    "DEFAULT_LEVERS",
    "DEFAULT_FEATURE_NAMES",
    "DEFAULT_SIGNATURE_DIM",
    "DEFAULT_PRE_ONSET_QUARTERS",
    "DEFAULT_POST_ONSET_QUARTERS",
    "N_MACRO_FEATURES",
    "LIBRARY_MANIFEST_NAME",
    "CASE_FILE_SUFFIX",
    # Exceptions
    "CaseMemoryError",
    "CaseSchemaError",
    "CaseLibraryError",
    "EncoderNotFittedError",
    # Schema objects
    "PolicyAction",
    "AuthoritySnapshot",
    "CrisisCase",
    "CaseLibrary",
    # Encoder
    "CaseSignatureEncoder",
    # Helpers
    "build_signature_matrix",
    "supcon_loss",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Semantic version of the on-disk case schema. The major component is
#: incremented on any change that breaks deserialisation of existing
#: ``.case.json`` files; the minor component for backward-compatible
#: additions; the patch component for documentation-only changes.
#: Loading a case with a major-version mismatch raises CaseSchemaError;
#: loading with a minor/patch mismatch emits a warning and proceeds.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: Canonical Laeven-Valencia crisis-type vocabulary. The "none" label is
#: deliberately absent: this library stores only *crises*. Coordinator-
#: level classification uses a separate label set that includes "none".
CRISIS_TYPES: Final[tuple[str, ...]] = (
    "banking",
    "currency",
    "sovereign",
    "twin",
    "triple",
)

#: Canonical policymaking-institution vocabulary. Authority-graph
#: snapshots inside cases must use exactly these labels. To add an
#: institution (e.g. a deposit-insurance agency), extend this constant
#: AND bump SCHEMA_VERSION to MAJOR.minor+1.0.
INSTITUTIONS: Final[tuple[str, ...]] = (
    "central_bank",
    "financial_supervisor",
    "ministry_of_finance",
)

#: Canonical policy-lever vocabulary. The same eight levers used by
#: ``mitigation.py`` AuthorityGraph; we replicate them here so the case
#: schema is self-contained.
DEFAULT_LEVERS: Final[tuple[str, ...]] = (
    "policy_rate",
    "fx_intervention",
    "reserve_requirement",
    "capital_adequacy_ratio",
    "loan_to_value_cap",
    "countercyclical_buffer",
    "fiscal_stance",
    "debt_issuance",
)

#: Canonical macro-feature ordering for the trajectory tensors. Cases
#: produced by ``data_pipeline.py`` use exactly this ordering; any
#: deviation raises CaseSchemaError at load time. Seven macro state
#: variables + five engineered features = 12 columns.
DEFAULT_FEATURE_NAMES: Final[tuple[str, ...]] = (
    # 7 macro state variables
    "output_gap",
    "inflation_yoy",
    "reer_log_dev",
    "credit_gap",
    "equity_returns_qoq",
    "sovereign_spread_bp",
    "fx_reserves_months",
    # 5 engineered features
    "inflation_yoy_change",
    "real_gdp_growth_yoy",
    "real_broad_money_growth_yoy",
    "real_credit_growth",
    "output_gap_squared",
)

#: Width of the macro feature panel. Must equal len(DEFAULT_FEATURE_NAMES).
N_MACRO_FEATURES: Final[int] = len(DEFAULT_FEATURE_NAMES)

#: Default dimensionality of the signature space. 32 matches the
#: coordinator's disagreement embedding for downstream compatibility.
DEFAULT_SIGNATURE_DIM: Final[int] = 32

#: Default length of the pre-onset trajectory window (in quarters). The
#: 20-quarter window matches Abadie (2021) recommended pre-treatment
#: fitting horizon for synthetic-control studies.
DEFAULT_PRE_ONSET_QUARTERS: Final[int] = 20

#: Default length of the post-onset realised-outcome window. The
#: 12-quarter window matches the canonical Laeven-Valencia output-loss
#: measurement.
DEFAULT_POST_ONSET_QUARTERS: Final[int] = 12

#: Minimum allowed pre-onset trajectory length. Cases with fewer than
#: 8 quarters of pre-onset data are rejected; the signature encoder
#: cannot reliably embed shorter trajectories.
MIN_PRE_ONSET_QUARTERS: Final[int] = 8

#: Manifest file name in the library directory.
LIBRARY_MANIFEST_NAME: Final[str] = "library_manifest.json"

#: Per-case file suffix. The convention ``ISO3_YYYY_Qn.case.json`` keeps
#: directory listings sorted by country then chronology when viewed in a
#: file browser.
CASE_FILE_SUFFIX: Final[str] = ".case.json"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class CaseMemoryError(Exception):
    """Base class for all case-memory exceptions."""


class CaseSchemaError(CaseMemoryError):
    """A case fails schema validation (wrong shape, unknown vocabulary
    item, version mismatch, malformed timeline, etc.)."""


class CaseLibraryError(CaseMemoryError):
    """A library-level operation fails (duplicate case_id, manifest
    corruption, checksum mismatch, empty library where one is required)."""


class EncoderNotFittedError(CaseMemoryError):
    """The signature encoder is used in inference mode before .fit()
    has been called. Raised by ``encode`` only if the caller requested
    strict-fitted behaviour; the default is a one-time warning."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    """Serialise to canonical JSON: sorted keys, compact separators,
    ASCII-escaped non-ASCII characters. Produces byte-stable output
    across platforms for any combination of dicts, lists, strings,
    ints, floats, and bools."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(data: Union[str, bytes]) -> str:
    """Hex-digest SHA-256 of a string or bytes input. Strings are
    encoded as UTF-8 before hashing."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text to ``path`` atomically: write to ``path.tmp`` first,
    then ``os.replace``. A crash mid-write leaves the original file
    intact rather than producing a half-written file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def _utc_now_iso() -> str:
    """ISO 8601 timestamp in UTC, second precision. Stable format for
    JSON serialisation."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _check_compatible_version(
    found_version: str,
    expected_version: str = SCHEMA_VERSION,
    context: str = "object",
) -> None:
    """Verify that ``found_version`` is compatible with ``expected_version``.

    Compatibility policy
    --------------------
    * MAJOR mismatch  -> CaseSchemaError (cannot deserialise safely).
    * MINOR mismatch  -> logger.warning (additive change; default
      values applied to missing fields).
    * PATCH mismatch  -> silent (docs-only change).
    """
    try:
        found = tuple(int(p) for p in found_version.split("."))
        expected = tuple(int(p) for p in expected_version.split("."))
    except (ValueError, AttributeError) as exc:
        raise CaseSchemaError(
            f"{context}: malformed schema_version {found_version!r}; "
            f"expected MAJOR.MINOR.PATCH"
        ) from exc
    if len(found) != 3 or len(expected) != 3:
        raise CaseSchemaError(
            f"{context}: schema_version must be MAJOR.MINOR.PATCH; "
            f"got {found_version!r}"
        )
    if found[0] != expected[0]:
        raise CaseSchemaError(
            f"{context}: schema major-version mismatch (found "
            f"{found_version!r}, code supports {expected_version!r}). "
            f"Re-extract the case library or run a migration."
        )
    if found[1] != expected[1]:
        logger.warning(
            "%s: schema minor-version mismatch (found %s, code supports %s); "
            "loading with default values for any new fields.",
            context, found_version, expected_version,
        )


def _validate_iso3(s: str, context: str = "country_iso3") -> None:
    if not isinstance(s, str) or len(s) != 3 or not s.isupper() or not s.isalpha():
        raise CaseSchemaError(
            f"{context}: must be a 3-letter uppercase ISO-3166-alpha-3 "
            f"code; got {s!r}"
        )


def _validate_iso_date(s: str, context: str = "date") -> None:
    try:
        datetime.fromisoformat(s)
    except (TypeError, ValueError) as exc:
        raise CaseSchemaError(
            f"{context}: invalid ISO 8601 date {s!r}"
        ) from exc


def _validate_quarter(q: int, context: str = "quarter") -> None:
    if not isinstance(q, int) or q not in (1, 2, 3, 4):
        raise CaseSchemaError(
            f"{context}: must be int in {{1,2,3,4}}; got {q!r}"
        )


def _array_to_jsonable(arr: np.ndarray) -> list[list[float]]:
    """Convert a 2-D numpy array to nested Python lists of native
    floats, suitable for JSON serialisation. Python floats have a
    platform-stable repr in Python 3, so the resulting JSON is
    byte-stable across operating systems."""
    if arr.ndim != 2:
        raise CaseSchemaError(
            f"trajectory must be 2-D (timesteps x features); got shape {arr.shape}"
        )
    return arr.astype(float, copy=False).tolist()


def _jsonable_to_array(lst: Sequence[Sequence[float]], context: str) -> np.ndarray:
    """Inverse of ``_array_to_jsonable``. Validates rectangularity."""
    try:
        arr = np.asarray(lst, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise CaseSchemaError(
            f"{context}: could not parse trajectory matrix"
        ) from exc
    if arr.ndim != 2:
        raise CaseSchemaError(
            f"{context}: trajectory must be 2-D; got ndim={arr.ndim}"
        )
    return arr


# ---------------------------------------------------------------------------
# PolicyAction: one timestamped row of a case's policy timeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicyAction:
    """One policy action in a case's chronological timeline.

    Parameters
    ----------
    date : str
        ISO 8601 date (YYYY-MM-DD) on which the action was announced.
    institution : str
        One of :data:`INSTITUTIONS`.
    lever : str
        One of :data:`DEFAULT_LEVERS`.
    value : float
        Signed magnitude of the action in ``units``. Positive values
        denote tightening (e.g. +625 bp policy rate hike); negative
        values denote loosening.
    units : str
        Units of ``value``. Common choices: ``"bp_change"`` for rate
        changes, ``"pp_of_gdp"`` for fiscal-stance changes,
        ``"usd_billion"`` for FX interventions, ``"binary"`` for
        on/off levers (e.g. IMF program signed).
    note : str, optional
        Free-text annotation, e.g. ``"emergency 8pm announcement"``.
        Empty by default.

    Notes
    -----
    Equality and hashing use all five required fields plus the note.
    Two actions are distinct if they differ in any way; this matters
    when a case includes coordinated same-day actions across multiple
    institutions.
    """

    date: str
    institution: str
    lever: str
    value: float
    units: str
    note: str = ""

    def __post_init__(self) -> None:
        _validate_iso_date(self.date, context="PolicyAction.date")
        if self.institution not in INSTITUTIONS:
            raise CaseSchemaError(
                f"PolicyAction.institution: unknown institution "
                f"{self.institution!r}; allowed {list(INSTITUTIONS)}"
            )
        if self.lever not in DEFAULT_LEVERS:
            raise CaseSchemaError(
                f"PolicyAction.lever: unknown lever {self.lever!r}; "
                f"allowed {list(DEFAULT_LEVERS)}"
            )
        if not isinstance(self.value, (int, float)) or not np.isfinite(self.value):
            raise CaseSchemaError(
                f"PolicyAction.value must be a finite number; got {self.value!r}"
            )
        if not isinstance(self.units, str) or not self.units:
            raise CaseSchemaError(
                f"PolicyAction.units must be a non-empty string; got {self.units!r}"
            )
        # Coerce value to float (frozen dataclass; use object.__setattr__)
        object.__setattr__(self, "value", float(self.value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "institution": self.institution,
            "lever": self.lever,
            "value": self.value,
            "units": self.units,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PolicyAction":
        try:
            return cls(
                date=str(d["date"]),
                institution=str(d["institution"]),
                lever=str(d["lever"]),
                value=float(d["value"]),
                units=str(d["units"]),
                note=str(d.get("note", "")),
            )
        except KeyError as exc:
            raise CaseSchemaError(
                f"PolicyAction.from_dict: missing required field {exc.args[0]!r}"
            ) from exc


# ---------------------------------------------------------------------------
# AuthoritySnapshot: institution -> levers mapping at the time of the case
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuthoritySnapshot:
    """The policy-authority graph in force at the time of a case.

    Stored as an institution -> frozenset of levers mapping. Validation
    guarantees that every lever is owned by exactly one institution
    (no double-ownership), that every institution is in
    :data:`INSTITUTIONS`, and that every lever is in
    :data:`DEFAULT_LEVERS`. Cases with partial coverage (e.g. a
    capital-controls regime in which one lever was unassigned to any
    institution) are permitted: lever coverage need not be exhaustive.

    Parameters
    ----------
    institution_levers : Mapping[str, Iterable[str]]
        Maps each institution to the levers it controlled. Order of
        levers inside each value is irrelevant; the canonical form is
        a sorted tuple.
    """

    institution_levers: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.institution_levers, Mapping):
            raise CaseSchemaError(
                f"AuthoritySnapshot.institution_levers must be a mapping; "
                f"got {type(self.institution_levers).__name__}"
            )
        canonical: dict[str, tuple[str, ...]] = {}
        seen_levers: set[str] = set()
        for inst, levers in self.institution_levers.items():
            if inst not in INSTITUTIONS:
                raise CaseSchemaError(
                    f"AuthoritySnapshot: unknown institution {inst!r}; "
                    f"allowed {list(INSTITUTIONS)}"
                )
            lever_tuple: tuple[str, ...] = tuple(sorted(set(levers)))
            for lever in lever_tuple:
                if lever not in DEFAULT_LEVERS:
                    raise CaseSchemaError(
                        f"AuthoritySnapshot: unknown lever {lever!r} under "
                        f"institution {inst!r}; allowed {list(DEFAULT_LEVERS)}"
                    )
                if lever in seen_levers:
                    raise CaseSchemaError(
                        f"AuthoritySnapshot: lever {lever!r} is controlled "
                        f"by more than one institution; each lever must "
                        f"have exactly one owner"
                    )
                seen_levers.add(lever)
            canonical[inst] = lever_tuple
        # Replace with canonical (sorted) form
        object.__setattr__(self, "institution_levers", canonical)

    @property
    def covered_levers(self) -> tuple[str, ...]:
        """Sorted tuple of every lever covered by the snapshot
        (across all institutions)."""
        return tuple(sorted({lv for vs in self.institution_levers.values() for lv in vs}))

    def to_dict(self) -> dict[str, list[str]]:
        return {inst: list(levers) for inst, levers in self.institution_levers.items()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Sequence[str]]) -> "AuthoritySnapshot":
        return cls(institution_levers={k: tuple(v) for k, v in d.items()})

    @classmethod
    def default(cls) -> "AuthoritySnapshot":
        """The modal authority architecture observed across the 28-country
        panel: the central bank owns rate / FX / reserve requirement,
        the financial supervisor owns CAR / LTV / countercyclical buffer,
        and the ministry of finance owns fiscal stance / debt issuance."""
        return cls(institution_levers={
            "central_bank": (
                "fx_intervention", "policy_rate", "reserve_requirement",
            ),
            "financial_supervisor": (
                "capital_adequacy_ratio", "countercyclical_buffer",
                "loan_to_value_cap",
            ),
            "ministry_of_finance": (
                "debt_issuance", "fiscal_stance",
            ),
        })


# ---------------------------------------------------------------------------
# CrisisCase: the immutable historical fact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrisisCase:
    """One historical emerging-market crisis case.

    A frozen, validated, content-hashable record of a single crisis
    episode. The fields fall into four groups:

    Identification
        ``case_id``, ``country_iso3``, ``onset_year``,
        ``onset_quarter``. The case_id by convention is
        ``f"{country_iso3}_{onset_year}_Q{onset_quarter}"``; the
        constructor validates this convention.

    Classification
        ``crisis_type`` from :data:`CRISIS_TYPES`.

    Time series and policy
        ``pre_onset_trajectory`` (numpy array of shape
        ``(T_pre, N_MACRO_FEATURES)``), ``post_onset_trajectory``
        (numpy array of shape ``(T_post, N_MACRO_FEATURES)``),
        ``feature_names`` (column ordering, must equal
        :data:`DEFAULT_FEATURE_NAMES` by default),
        ``policy_timeline`` (sorted chronologically), and
        ``authority_snapshot``.

    Outcome and metadata
        ``output_loss_cumulative_gdp`` (in percent of pre-onset
        cumulative GDP over the eval horizon), ``provenance``
        (list of strings citing primary sources for the label, the
        timeline, and the loss measurement), ``data_quality`` in
        ``[0, 1]``, optional ``notes``, optional ``end_year`` /
        ``end_quarter`` for closed crises (None for ongoing cases
        like Lebanon), the ``schema_version``, and the ``created_at``
        UTC timestamp.

    Immutability and arrays
    -----------------------
    The dataclass is frozen, so its field bindings cannot be reassigned.
    The numpy arrays stored as fields are additionally marked
    ``writeable=False`` in ``__post_init__``; attempting to mutate them
    raises ``ValueError`` at the numpy level. This matters because cases
    are often shared by reference across many retrieval calls; defensive
    copies would be wasteful but undisciplined mutation would corrupt
    the library.

    Equality
    --------
    Two cases are equal if their canonical-JSON serialisations are
    equal. ``content_checksum()`` returns the SHA-256 of that JSON,
    suitable for change detection.
    """

    case_id: str
    country_iso3: str
    onset_year: int
    onset_quarter: int
    crisis_type: str
    pre_onset_trajectory: np.ndarray
    post_onset_trajectory: np.ndarray
    feature_names: tuple[str, ...]
    policy_timeline: tuple[PolicyAction, ...]
    authority_snapshot: AuthoritySnapshot
    output_loss_cumulative_gdp: float
    provenance: tuple[str, ...]
    data_quality: float
    end_year: Optional[int] = None
    end_quarter: Optional[int] = None
    notes: str = ""
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=_utc_now_iso)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        _check_compatible_version(self.schema_version, context=f"CrisisCase[{self.case_id}]")

        # Identification
        if not isinstance(self.case_id, str) or not self.case_id:
            raise CaseSchemaError(
                f"CrisisCase.case_id must be a non-empty string; got {self.case_id!r}"
            )
        _validate_iso3(self.country_iso3, context=f"CrisisCase[{self.case_id}].country_iso3")
        if not isinstance(self.onset_year, int) or not (1900 <= self.onset_year <= 2100):
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].onset_year must be int in "
                f"[1900, 2100]; got {self.onset_year!r}"
            )
        _validate_quarter(self.onset_quarter, context=f"CrisisCase[{self.case_id}].onset_quarter")

        # case_id convention check (warn rather than reject; legacy cases
        # may have legitimate non-conforming ids)
        expected_id = f"{self.country_iso3}_{self.onset_year}_Q{self.onset_quarter}"
        if self.case_id != expected_id:
            logger.warning(
                "CrisisCase.case_id %r does not match the canonical "
                "convention %r; this is allowed but unusual.",
                self.case_id, expected_id,
            )

        # Classification
        if self.crisis_type not in CRISIS_TYPES:
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].crisis_type: unknown type "
                f"{self.crisis_type!r}; allowed {list(CRISIS_TYPES)}"
            )

        # Feature names
        if (not isinstance(self.feature_names, tuple)
                or len(self.feature_names) == 0):
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].feature_names must be a "
                f"non-empty tuple"
            )

        # Trajectory shapes
        for name, arr in (
            ("pre_onset_trajectory", self.pre_onset_trajectory),
            ("post_onset_trajectory", self.post_onset_trajectory),
        ):
            if not isinstance(arr, np.ndarray):
                raise CaseSchemaError(
                    f"CrisisCase[{self.case_id}].{name} must be a numpy "
                    f"array; got {type(arr).__name__}"
                )
            if arr.ndim != 2:
                raise CaseSchemaError(
                    f"CrisisCase[{self.case_id}].{name} must be 2-D; "
                    f"got shape {arr.shape}"
                )
            if arr.shape[1] != len(self.feature_names):
                raise CaseSchemaError(
                    f"CrisisCase[{self.case_id}].{name}: width "
                    f"{arr.shape[1]} does not match feature_names length "
                    f"{len(self.feature_names)}"
                )
            if not np.all(np.isfinite(arr)):
                raise CaseSchemaError(
                    f"CrisisCase[{self.case_id}].{name} contains non-finite "
                    f"values (NaN or inf)"
                )

        if self.pre_onset_trajectory.shape[0] < MIN_PRE_ONSET_QUARTERS:
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].pre_onset_trajectory: only "
                f"{self.pre_onset_trajectory.shape[0]} quarters; need "
                f">= {MIN_PRE_ONSET_QUARTERS}"
            )
        if self.post_onset_trajectory.shape[0] < 1:
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].post_onset_trajectory must "
                f"have at least one quarter"
            )

        # Mark arrays read-only (in place, no rebinding required)
        self.pre_onset_trajectory.setflags(write=False)
        self.post_onset_trajectory.setflags(write=False)

        # Policy timeline
        if not isinstance(self.policy_timeline, tuple):
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].policy_timeline must be a "
                f"tuple; got {type(self.policy_timeline).__name__}"
            )
        for i, a in enumerate(self.policy_timeline):
            if not isinstance(a, PolicyAction):
                raise CaseSchemaError(
                    f"CrisisCase[{self.case_id}].policy_timeline[{i}] "
                    f"must be a PolicyAction; got {type(a).__name__}"
                )
        # Ensure chronological order (sort if not already)
        sorted_timeline = tuple(sorted(self.policy_timeline, key=lambda a: a.date))
        if sorted_timeline != self.policy_timeline:
            object.__setattr__(self, "policy_timeline", sorted_timeline)

        # Authority snapshot
        if not isinstance(self.authority_snapshot, AuthoritySnapshot):
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].authority_snapshot must be "
                f"an AuthoritySnapshot; got {type(self.authority_snapshot).__name__}"
            )

        # Outcome
        if (not isinstance(self.output_loss_cumulative_gdp, (int, float))
                or not np.isfinite(self.output_loss_cumulative_gdp)):
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].output_loss_cumulative_gdp "
                f"must be a finite number; got "
                f"{self.output_loss_cumulative_gdp!r}"
            )
        # Output loss is expressed in percent of GDP; negative means
        # the country avoided a loss (rare but allowed for placebo
        # cases). Cap absolute value at 200% for sanity.
        if abs(self.output_loss_cumulative_gdp) > 200.0:
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].output_loss_cumulative_gdp "
                f"= {self.output_loss_cumulative_gdp}; absolute value > 200 "
                f"is implausible"
            )

        # Provenance and data quality
        if (not isinstance(self.provenance, tuple)
                or any(not isinstance(p, str) or not p for p in self.provenance)):
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].provenance must be a tuple "
                f"of non-empty strings"
            )
        if len(self.provenance) == 0:
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].provenance must contain at "
                f"least one citation"
            )
        if not (0.0 <= self.data_quality <= 1.0):
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].data_quality must be in "
                f"[0, 1]; got {self.data_quality}"
            )

        # Optional end date
        if (self.end_year is None) != (self.end_quarter is None):
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}]: end_year and end_quarter "
                f"must both be set or both be None"
            )
        if self.end_year is not None:
            if not (1900 <= self.end_year <= 2100):
                raise CaseSchemaError(
                    f"CrisisCase[{self.case_id}].end_year out of range"
                )
            _validate_quarter(
                self.end_quarter, context=f"CrisisCase[{self.case_id}].end_quarter"
            )
            # Check end >= onset
            end_idx = self.end_year * 4 + self.end_quarter
            onset_idx = self.onset_year * 4 + self.onset_quarter
            if end_idx < onset_idx:
                raise CaseSchemaError(
                    f"CrisisCase[{self.case_id}]: end "
                    f"({self.end_year}Q{self.end_quarter}) precedes "
                    f"onset ({self.onset_year}Q{self.onset_quarter})"
                )

        # Notes
        if not isinstance(self.notes, str):
            raise CaseSchemaError(
                f"CrisisCase[{self.case_id}].notes must be a string"
            )

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def onset_date(self) -> pd.Timestamp:
        """End-of-quarter Timestamp for the crisis onset."""
        month = self.onset_quarter * 3
        return pd.Timestamp(year=self.onset_year, month=month, day=1) \
            + pd.offsets.MonthEnd(0)

    @property
    def is_ongoing(self) -> bool:
        """True if the case has no recorded end (e.g. Lebanon 2019-)."""
        return self.end_year is None

    @property
    def n_pre_onset_quarters(self) -> int:
        return int(self.pre_onset_trajectory.shape[0])

    @property
    def n_post_onset_quarters(self) -> int:
        return int(self.post_onset_trajectory.shape[0])

    def macro_signature_raw(self) -> np.ndarray:
        """Flatten the pre-onset trajectory to a single 1-D vector,
        in row-major (time-major) order. This is the input to the
        signature encoder."""
        return np.ascontiguousarray(self.pre_onset_trajectory.copy())

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation. All numpy
        arrays are converted to nested lists of Python floats."""
        return {
            "case_id": self.case_id,
            "country_iso3": self.country_iso3,
            "onset_year": int(self.onset_year),
            "onset_quarter": int(self.onset_quarter),
            "end_year": (None if self.end_year is None else int(self.end_year)),
            "end_quarter": (None if self.end_quarter is None
                            else int(self.end_quarter)),
            "crisis_type": self.crisis_type,
            "feature_names": list(self.feature_names),
            "pre_onset_trajectory": _array_to_jsonable(self.pre_onset_trajectory),
            "post_onset_trajectory": _array_to_jsonable(self.post_onset_trajectory),
            "policy_timeline": [a.to_dict() for a in self.policy_timeline],
            "authority_snapshot": self.authority_snapshot.to_dict(),
            "output_loss_cumulative_gdp": float(self.output_loss_cumulative_gdp),
            "provenance": list(self.provenance),
            "data_quality": float(self.data_quality),
            "notes": self.notes,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialise to a JSON string. ``indent=None`` produces canonical
        (compact) JSON suitable for hashing; the default ``indent=2``
        produces a human-readable form suitable for storage."""
        if indent is None:
            return _canonical_json(self.to_dict())
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=indent,
            ensure_ascii=True, allow_nan=False,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CrisisCase":
        """Construct a CrisisCase from a dict produced by ``to_dict``."""
        if "schema_version" in d:
            _check_compatible_version(
                d["schema_version"],
                context=f"CrisisCase[{d.get('case_id', '?')}]",
            )
        try:
            pre = _jsonable_to_array(
                d["pre_onset_trajectory"],
                context=f"CrisisCase[{d.get('case_id', '?')}].pre_onset_trajectory",
            )
            post = _jsonable_to_array(
                d["post_onset_trajectory"],
                context=f"CrisisCase[{d.get('case_id', '?')}].post_onset_trajectory",
            )
            timeline = tuple(
                PolicyAction.from_dict(a) for a in d.get("policy_timeline", ())
            )
            authority = AuthoritySnapshot.from_dict(d["authority_snapshot"])
            return cls(
                case_id=str(d["case_id"]),
                country_iso3=str(d["country_iso3"]),
                onset_year=int(d["onset_year"]),
                onset_quarter=int(d["onset_quarter"]),
                crisis_type=str(d["crisis_type"]),
                pre_onset_trajectory=pre,
                post_onset_trajectory=post,
                feature_names=tuple(d["feature_names"]),
                policy_timeline=timeline,
                authority_snapshot=authority,
                output_loss_cumulative_gdp=float(d["output_loss_cumulative_gdp"]),
                provenance=tuple(d["provenance"]),
                data_quality=float(d["data_quality"]),
                end_year=(None if d.get("end_year") is None
                          else int(d["end_year"])),
                end_quarter=(None if d.get("end_quarter") is None
                             else int(d["end_quarter"])),
                notes=str(d.get("notes", "")),
                schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
                created_at=str(d.get("created_at", _utc_now_iso())),
            )
        except KeyError as exc:
            raise CaseSchemaError(
                f"CrisisCase.from_dict: missing required field {exc.args[0]!r} "
                f"in case {d.get('case_id', '?')!r}"
            ) from exc

    @classmethod
    def from_json(cls, s: str) -> "CrisisCase":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------ #
    # Hashing / equality
    # ------------------------------------------------------------------ #

    def content_checksum(self) -> str:
        """SHA-256 hex digest of the canonical-JSON serialisation.

        Two cases produce the same checksum iff they would serialise
        to the same canonical JSON, i.e. their content is identical
        modulo ordering of dict keys. The created_at field IS included
        in the checksum so that two cases extracted from the same data
        at different times remain distinguishable.
        """
        return _sha256_hex(self.to_json(indent=None))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CrisisCase):
            return NotImplemented
        return self.content_checksum() == other.content_checksum()

    def __hash__(self) -> int:
        return int(self.content_checksum()[:16], 16)


# ---------------------------------------------------------------------------
# CaseLibrary: a directory-backed mutable collection of cases
# ---------------------------------------------------------------------------


class CaseLibrary:
    """A versioned, content-checksummed collection of CrisisCases.

    On disk, a library is a directory containing one ``.case.json``
    file per case plus a ``library_manifest.json`` manifest that lists
    every case_id, its content checksum, and the library-level checksum
    computed over the sorted (case_id, content_checksum) pairs. The
    library checksum is invariant to ordering on disk (we sort before
    hashing) but changes whenever any case is added, removed, or
    modified.

    Construction
    ------------
    >>> lib = CaseLibrary()                       # empty
    >>> lib.add(case)                             # add one
    >>> lib.save(Path("library/"))                # persist
    >>> lib2 = CaseLibrary.load(Path("library/")) # reload

    Querying
    --------
    >>> lib.query(country="ARG")                  # all ARG cases
    >>> lib.query(crisis_type="triple")           # all triple crises
    >>> lib.query(onset_year_range=(2010, 2024))  # date filter
    >>> tur_2018 = lib["TUR_2018_Q3"]             # by case_id

    Splitting
    ---------
    >>> train, test = lib.split_by_year(2018)     # for hold-out eval
    """

    def __init__(
        self,
        cases: Optional[Iterable[CrisisCase]] = None,
        *,
        schema_version: str = SCHEMA_VERSION,
        provenance: Sequence[str] = (),
        notes: str = "",
    ) -> None:
        self._cases: dict[str, CrisisCase] = {}
        self.schema_version: str = schema_version
        self.provenance: tuple[str, ...] = tuple(provenance)
        self.notes: str = notes
        self.created_at: str = _utc_now_iso()
        if cases is not None:
            for c in cases:
                self.add(c)

    # ------------------------------------------------------------------ #
    # Container protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self) -> Iterator[str]:
        """Iterate over case_ids in sorted order (deterministic)."""
        return iter(sorted(self._cases))

    def __contains__(self, case_id: object) -> bool:
        return isinstance(case_id, str) and case_id in self._cases

    def __getitem__(self, case_id: str) -> CrisisCase:
        try:
            return self._cases[case_id]
        except KeyError:
            raise CaseLibraryError(
                f"CaseLibrary: no case with id {case_id!r}; library has "
                f"{len(self._cases)} cases"
            ) from None

    def cases(self) -> tuple[CrisisCase, ...]:
        """All cases, ordered by case_id."""
        return tuple(self._cases[cid] for cid in iter(self))

    def case_ids(self) -> tuple[str, ...]:
        """All case_ids in sorted order."""
        return tuple(iter(self))

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def add(self, case: CrisisCase, *, overwrite: bool = False) -> None:
        """Add a case to the library.

        Raises CaseLibraryError if a case with the same id already
        exists, unless ``overwrite=True``. Overwriting silently is
        a footgun in collaborative settings, hence the explicit flag.
        """
        if not isinstance(case, CrisisCase):
            raise TypeError(
                f"CaseLibrary.add: expected CrisisCase, got "
                f"{type(case).__name__}"
            )
        if case.case_id in self._cases and not overwrite:
            raise CaseLibraryError(
                f"CaseLibrary.add: case_id {case.case_id!r} already "
                f"present; pass overwrite=True to replace"
            )
        self._cases[case.case_id] = case

    def remove(self, case_id: str) -> CrisisCase:
        """Remove a case and return it. Raises if not present."""
        try:
            return self._cases.pop(case_id)
        except KeyError:
            raise CaseLibraryError(
                f"CaseLibrary.remove: no case with id {case_id!r}"
            ) from None

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def query(
        self,
        *,
        country: Optional[str] = None,
        crisis_type: Optional[str] = None,
        onset_year_range: Optional[tuple[int, int]] = None,
        min_data_quality: Optional[float] = None,
    ) -> tuple[CrisisCase, ...]:
        """Return cases matching all of the supplied filters.

        Parameters
        ----------
        country : str, optional
            ISO-3 country code; case-sensitive.
        crisis_type : str, optional
            One of :data:`CRISIS_TYPES`.
        onset_year_range : (int, int), optional
            Inclusive ``(start_year, end_year)``.
        min_data_quality : float in [0, 1], optional
            Drop cases whose ``data_quality`` is strictly below this
            threshold.

        Returns
        -------
        tuple of CrisisCase, sorted by case_id.
        """
        result: list[CrisisCase] = []
        for cid in iter(self):
            c = self._cases[cid]
            if country is not None and c.country_iso3 != country:
                continue
            if crisis_type is not None and c.crisis_type != crisis_type:
                continue
            if onset_year_range is not None:
                lo, hi = onset_year_range
                if not (lo <= c.onset_year <= hi):
                    continue
            if min_data_quality is not None and c.data_quality < min_data_quality:
                continue
            result.append(c)
        return tuple(result)

    def split_by_year(
        self, cutoff_year: int,
    ) -> tuple["CaseLibrary", "CaseLibrary"]:
        """Split into (train, test) by onset year. Cases with
        ``onset_year <= cutoff_year`` go into train; the rest into
        test. Preserves provenance and notes; bumps created_at."""
        train_cases = [c for c in self.cases() if c.onset_year <= cutoff_year]
        test_cases = [c for c in self.cases() if c.onset_year > cutoff_year]
        train = CaseLibrary(
            train_cases, schema_version=self.schema_version,
            provenance=self.provenance + (f"split_by_year<={cutoff_year}",),
            notes=self.notes,
        )
        test = CaseLibrary(
            test_cases, schema_version=self.schema_version,
            provenance=self.provenance + (f"split_by_year>{cutoff_year}",),
            notes=self.notes,
        )
        return train, test

    def crisis_type_counts(self) -> dict[str, int]:
        """Distribution of crisis types in the library."""
        counts: dict[str, int] = {t: 0 for t in CRISIS_TYPES}
        for c in self.cases():
            counts[c.crisis_type] = counts.get(c.crisis_type, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    # Checksum
    # ------------------------------------------------------------------ #

    def case_checksums(self) -> dict[str, str]:
        """Map case_id -> SHA-256 content checksum, sorted by case_id."""
        return {cid: self._cases[cid].content_checksum() for cid in iter(self)}

    def library_checksum(self) -> str:
        """SHA-256 over the sorted (case_id, case_checksum) pairs.
        Stable across permutations of internal ordering and across
        platforms."""
        pairs = sorted(self.case_checksums().items())
        return _sha256_hex(_canonical_json(pairs))

    # ------------------------------------------------------------------ #
    # Disk I/O
    # ------------------------------------------------------------------ #

    def save(self, path: Path, *, overwrite: bool = False) -> Path:
        """Save the library as a directory of per-case JSON files plus
        a manifest. The directory must either not exist or contain a
        compatible manifest; pass ``overwrite=True`` to replace any
        existing library in place.

        Returns
        -------
        Path
            The directory the library was saved to (same as ``path``).
        """
        path = Path(path)
        if path.exists():
            if not path.is_dir():
                raise CaseLibraryError(
                    f"CaseLibrary.save: {path} exists and is not a directory"
                )
            existing_manifest = path / LIBRARY_MANIFEST_NAME
            if existing_manifest.exists() and not overwrite:
                raise CaseLibraryError(
                    f"CaseLibrary.save: a library already exists at {path}; "
                    f"pass overwrite=True to replace"
                )
            if overwrite:
                # Remove any pre-existing .case.json files (don't touch
                # other files the user might have stashed in the dir).
                for f in path.glob(f"*{CASE_FILE_SUFFIX}"):
                    f.unlink()
                if existing_manifest.exists():
                    existing_manifest.unlink()
        else:
            path.mkdir(parents=True, exist_ok=False)

        # Write each case
        for cid in iter(self):
            case = self._cases[cid]
            case_path = path / f"{cid}{CASE_FILE_SUFFIX}"
            _atomic_write_text(case_path, case.to_json(indent=2))

        # Write manifest
        manifest = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "saved_at": _utc_now_iso(),
            "n_cases": len(self),
            "case_checksums": self.case_checksums(),
            "library_checksum": self.library_checksum(),
            "provenance": list(self.provenance),
            "notes": self.notes,
            "crisis_type_counts": self.crisis_type_counts(),
        }
        _atomic_write_text(
            path / LIBRARY_MANIFEST_NAME,
            json.dumps(manifest, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        logger.info(
            "CaseLibrary.save: wrote %d cases to %s (checksum %s)",
            len(self), path, manifest["library_checksum"][:12],
        )
        return path

    @classmethod
    def load(
        cls, path: Path,
        *,
        verify_checksums: bool = True,
        strict: bool = True,
    ) -> "CaseLibrary":
        """Load a library from a directory.

        Parameters
        ----------
        path : Path
            Directory containing ``library_manifest.json`` and one or
            more ``*.case.json`` files.
        verify_checksums : bool, default True
            If True, re-compute each case's SHA-256 and the library-
            level SHA-256 and check them against the manifest.
            Mismatches raise CaseLibraryError unless ``strict=False``.
        strict : bool, default True
            If False, demote checksum mismatches to warnings.
        """
        path = Path(path)
        if not path.is_dir():
            raise CaseLibraryError(f"CaseLibrary.load: not a directory: {path}")
        manifest_path = path / LIBRARY_MANIFEST_NAME
        if not manifest_path.is_file():
            raise CaseLibraryError(
                f"CaseLibrary.load: no manifest at {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CaseLibraryError(
                f"CaseLibrary.load: manifest is not valid JSON: {exc}"
            ) from exc

        schema_version = manifest.get("schema_version", SCHEMA_VERSION)
        _check_compatible_version(schema_version, context=f"library@{path}")

        lib = cls(
            schema_version=schema_version,
            provenance=tuple(manifest.get("provenance", ())),
            notes=manifest.get("notes", ""),
        )
        lib.created_at = manifest.get("created_at", _utc_now_iso())

        # Load each case mentioned in the manifest. Cases on disk but
        # absent from the manifest are ignored (treated as orphans).
        manifest_checksums: Mapping[str, str] = manifest.get("case_checksums", {})
        for case_id in sorted(manifest_checksums):
            case_path = path / f"{case_id}{CASE_FILE_SUFFIX}"
            if not case_path.is_file():
                raise CaseLibraryError(
                    f"CaseLibrary.load: manifest references {case_id} but "
                    f"{case_path} does not exist"
                )
            case_text = case_path.read_text(encoding="utf-8")
            try:
                case = CrisisCase.from_json(case_text)
            except CaseSchemaError as exc:
                raise CaseLibraryError(
                    f"CaseLibrary.load: failed to load {case_path}: {exc}"
                ) from exc
            lib.add(case)
            if verify_checksums:
                computed = case.content_checksum()
                expected = manifest_checksums[case_id]
                if computed != expected:
                    msg = (
                        f"CaseLibrary.load: checksum mismatch for "
                        f"{case_id} (computed {computed[:12]}, "
                        f"manifest {expected[:12]})"
                    )
                    if strict:
                        raise CaseLibraryError(msg)
                    warnings.warn(msg, stacklevel=2)

        if verify_checksums:
            computed_lib = lib.library_checksum()
            expected_lib = manifest.get("library_checksum")
            if expected_lib is not None and computed_lib != expected_lib:
                msg = (
                    f"CaseLibrary.load: library-level checksum mismatch "
                    f"(computed {computed_lib[:12]}, manifest "
                    f"{expected_lib[:12]})"
                )
                if strict:
                    raise CaseLibraryError(msg)
                warnings.warn(msg, stacklevel=2)

        logger.info(
            "CaseLibrary.load: loaded %d cases from %s", len(lib), path,
        )
        return lib

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (f"CaseLibrary(n_cases={len(self)}, "
                f"schema_version={self.schema_version!r}, "
                f"checksum={self.library_checksum()[:12]!r})")

    def summary(self) -> dict[str, Any]:
        """Human-readable summary suitable for logging or printing."""
        return {
            "n_cases": len(self),
            "schema_version": self.schema_version,
            "library_checksum": self.library_checksum(),
            "crisis_type_counts": self.crisis_type_counts(),
            "countries": sorted({c.country_iso3 for c in self.cases()}),
            "onset_year_range": (
                (min(c.onset_year for c in self.cases()),
                 max(c.onset_year for c in self.cases()))
                if len(self) > 0 else None
            ),
            "provenance": list(self.provenance),
        }


# ---------------------------------------------------------------------------
# Signature encoder: trajectory -> dense vector
# ---------------------------------------------------------------------------


def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020).

    Parameters
    ----------
    features : Tensor of shape (B, D)
        L2-normalised embeddings. The function does not re-normalise;
        callers are responsible for ``F.normalize`` before calling.
    labels : Tensor of shape (B,) with integer class indices.
    temperature : float, default 0.1
        Softmax temperature; lower values sharpen the contrast.

    Returns
    -------
    Scalar Tensor. Differentiable.

    Notes
    -----
    Anchors with no positives in the batch (i.e. classes of size 1)
    do not contribute to the loss. If no anchor has any positive,
    the loss is identically zero (gradient-free).
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D; got shape {tuple(features.shape)}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must be (B,) matching "
            f"features.shape[0]={features.shape[0]}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0; got {temperature}")

    device = features.device
    B = features.shape[0]
    # Pairwise scaled dot-products. Features are assumed L2-normalised,
    # so dot-product == cosine similarity.
    sim = (features @ features.T) / temperature  # (B, B)
    # Subtract per-row max for numerical stability of exp.
    sim = sim - sim.detach().max(dim=1, keepdim=True).values

    eye = torch.eye(B, device=device, dtype=features.dtype)
    # Mask: same label, not self
    labels_col = labels.view(-1, 1)
    pos_mask = (labels_col == labels_col.T).to(features.dtype) - eye
    pos_mask = pos_mask.clamp(min=0.0)
    # Mask: not self (denominator includes negatives + positives)
    denom_mask = 1.0 - eye

    exp_sim = torch.exp(sim) * denom_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))

    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if not bool(valid.any()):
        return torch.zeros((), device=device, dtype=features.dtype, requires_grad=True)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)
    mean_log_prob_pos = mean_log_prob_pos[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


class CaseSignatureEncoder(nn.Module):
    """Map a pre-onset macro trajectory to a unit-norm dense signature.

    Architecture (deliberately small; ~10 k parameters for the default
    configuration, sized to ~150 cases without overfitting):

    1. Transpose input (B, T, F) -> (B, F, T) for 1-D convolution.
    2. Two stacked Conv1d layers with kernel 3, padding 1, GELU
       activations, and dropout. Captures local 1-3 quarter patterns.
    3. Transpose back to (B, T, H) and LayerNorm over H.
    4. Attention-weighted pooling over time: a small MLP scores each
       timestep; softmax-normalised weights produce a single
       hidden-dim vector per case.
    5. Two-layer MLP head with GELU and dropout projects to
       ``signature_dim``.
    6. L2-normalise so dot-product gives cosine similarity.

    The encoder is trained contrastively (SupCon) on crisis_type, so
    same-type cases cluster in signature space and different-type cases
    push apart. The crisis_type-conditional retrieval metric in
    ``analogy_engine.py`` operates on signatures produced by this
    encoder; consequently, signatures with high SupCon training loss
    indicate cases the retriever will struggle on, providing a
    diagnostic signal.

    Reproducibility
    ---------------
    The ``seed`` parameter is consumed once at construction to seed the
    parameter initialisation. Training itself draws from a separate
    ``torch.Generator`` seeded with the same value, so calling
    ``encoder.fit`` twice in the same process with the same library
    produces bit-identical results.
    """

    def __init__(
        self,
        *,
        n_features: int = N_MACRO_FEATURES,
        n_timesteps: int = DEFAULT_PRE_ONSET_QUARTERS,
        signature_dim: int = DEFAULT_SIGNATURE_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1; got {n_features}")
        if n_timesteps < MIN_PRE_ONSET_QUARTERS:
            raise ValueError(
                f"n_timesteps must be >= {MIN_PRE_ONSET_QUARTERS}; "
                f"got {n_timesteps}"
            )
        if signature_dim < 2:
            raise ValueError(f"signature_dim must be >= 2; got {signature_dim}")
        if hidden_dim < signature_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) should be >= signature_dim "
                f"({signature_dim})"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")

        self.config: dict[str, Any] = {
            "n_features": int(n_features),
            "n_timesteps": int(n_timesteps),
            "signature_dim": int(signature_dim),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "seed": int(seed),
        }

        # Deterministic init via global torch RNG. We seed before module
        # construction so PyTorch's per-module default init is applied
        # under a known seed, and restore the caller's RNG state on exit
        # so we do not pollute it. PyTorch's defaults correctly handle
        # LayerNorm (weight=1, bias=0), Conv1d (Kaiming-uniform), and
        # Linear (Kaiming-uniform) — re-implementing them here would
        # have to special-case each layer type and is fragile.
        _saved_cpu_state = torch.get_rng_state()
        _saved_cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            self.conv1 = nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.norm = nn.LayerNorm(hidden_dim)
            self.attn_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, signature_dim),
            )
            self._dropout = nn.Dropout(dropout)
        finally:
            torch.set_rng_state(_saved_cpu_state)
            if _saved_cuda_states is not None:
                torch.cuda.set_rng_state_all(_saved_cuda_states)

        self._is_fitted: bool = False
        self._fit_history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ #
    # Forward / encode
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, T, F) -> (B, signature_dim), L2-normalised."""
        if x.ndim != 3:
            raise ValueError(
                f"CaseSignatureEncoder.forward: expected 3-D input "
                f"(B, T, F); got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.config["n_features"]:
            raise ValueError(
                f"CaseSignatureEncoder.forward: expected "
                f"{self.config['n_features']} features; got {x.shape[-1]}"
            )
        if x.shape[-2] != self.config["n_timesteps"]:
            raise ValueError(
                f"CaseSignatureEncoder.forward: expected "
                f"{self.config['n_timesteps']} timesteps; got {x.shape[-2]}"
            )
        # (B, T, F) -> (B, F, T) -> convs -> (B, H, T) -> (B, T, H)
        h = x.transpose(-1, -2)
        h = F.gelu(self.conv1(h))
        h = self._dropout(h)
        h = F.gelu(self.conv2(h))
        h = self._dropout(h)
        h = h.transpose(-1, -2)
        h = self.norm(h)
        # Attention pool over time
        logits = self.attn_pool(h).squeeze(-1)  # (B, T)
        weights = F.softmax(logits, dim=-1)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=-2)  # (B, H)
        # Head + normalise
        z = self.head(pooled)
        z = F.normalize(z, p=2, dim=-1)
        return z

    def encode(
        self,
        trajectories: Union[np.ndarray, torch.Tensor],
        *,
        strict_fitted: bool = False,
    ) -> np.ndarray:
        """Encode one or many trajectories. Single-trajectory input
        (shape ``(T, F)``) is accepted and returns a 1-D signature
        of shape ``(D,)``; batched input (shape ``(B, T, F)``) returns
        a 2-D matrix of shape ``(B, D)``.

        ``strict_fitted=True`` raises EncoderNotFittedError if the
        encoder has not been .fit; the default emits a one-time
        warning and proceeds (producing random-projection signatures
        — useful for smoke tests, dangerous in production)."""
        if not self._is_fitted:
            if strict_fitted:
                raise EncoderNotFittedError(
                    "CaseSignatureEncoder.encode called before .fit"
                )
            warnings.warn(
                "CaseSignatureEncoder.encode called on an unfitted encoder; "
                "returned signatures are random projections. Call .fit first.",
                stacklevel=2,
            )

        single = False
        if isinstance(trajectories, np.ndarray):
            if trajectories.ndim == 2:
                trajectories = trajectories[np.newaxis, :, :]
                single = True
            x = torch.as_tensor(trajectories, dtype=torch.float32)
        elif isinstance(trajectories, torch.Tensor):
            if trajectories.ndim == 2:
                trajectories = trajectories.unsqueeze(0)
                single = True
            x = trajectories.to(dtype=torch.float32)
        else:
            raise TypeError(
                f"CaseSignatureEncoder.encode: trajectories must be "
                f"numpy.ndarray or torch.Tensor; got {type(trajectories).__name__}"
            )

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                z = self.forward(x)
        finally:
            if was_training:
                self.train()

        out = z.detach().cpu().numpy()
        return out[0] if single else out

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(
        self,
        library: CaseLibrary,
        *,
        n_epochs: int = 200,
        batch_size: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 0.1,
        grad_clip: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train the encoder contrastively on the library's crisis types.

        Parameters
        ----------
        library : CaseLibrary
            Source of (trajectory, crisis_type) pairs. Must contain at
            least 4 cases and at least one class with >= 2 cases (to
            form positive pairs).
        n_epochs : int, default 200
        batch_size : int or None
            If None (default), use full-batch SupCon — sensible for the
            small libraries we expect (~150 cases). Otherwise sample
            mini-batches of size ``batch_size`` with replacement to
            preserve class coverage.
        lr, weight_decay : AdamW hyperparameters.
        temperature : SupCon temperature.
        grad_clip : float, default 1.0
            Global gradient-norm clip.
        device : torch device.
        verbose : bool, default False
            If True, log a progress line every 20 epochs.

        Returns
        -------
        dict
            Training history with keys ``"epoch"`` and ``"loss"``.
        """
        if len(library) < 4:
            raise CaseLibraryError(
                f"CaseSignatureEncoder.fit: library has only "
                f"{len(library)} cases; need >= 4 for contrastive training"
            )
        # Check that at least one class has >= 2 examples
        counts = library.crisis_type_counts()
        n_positive_classes = sum(1 for v in counts.values() if v >= 2)
        if n_positive_classes == 0:
            raise CaseLibraryError(
                "CaseSignatureEncoder.fit: no crisis type has >= 2 cases; "
                "SupCon needs at least one class with positive pairs"
            )
        # Warn about small classes (still contributes to negatives)
        small_classes = {k: v for k, v in counts.items() if 0 < v < 2}
        if small_classes:
            warnings.warn(
                f"CaseSignatureEncoder.fit: crisis types with <2 cases "
                f"contribute no positive pairs: {small_classes}. These "
                f"cases enter only the negative pool.",
                stacklevel=2,
            )

        case_ids = library.case_ids()
        trajs = np.stack([
            library[cid].pre_onset_trajectory for cid in case_ids
        ], axis=0).astype(np.float32, copy=False)
        labels = np.array([
            CRISIS_TYPES.index(library[cid].crisis_type) for cid in case_ids
        ], dtype=np.int64)

        if trajs.shape[1] != self.config["n_timesteps"]:
            raise CaseSchemaError(
                f"CaseSignatureEncoder.fit: library trajectory length "
                f"{trajs.shape[1]} does not match encoder n_timesteps "
                f"{self.config['n_timesteps']}"
            )
        if trajs.shape[2] != self.config["n_features"]:
            raise CaseSchemaError(
                f"CaseSignatureEncoder.fit: library trajectory width "
                f"{trajs.shape[2]} does not match encoder n_features "
                f"{self.config['n_features']}"
            )

        device_ = torch.device(device)
        x_full = torch.as_tensor(trajs, device=device_)
        y_full = torch.as_tensor(labels, device=device_)
        n = x_full.shape[0]
        if batch_size is None or batch_size >= n:
            batch_size = n

        self.to(device_)
        self.train()
        optim = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay,
        )

        gen = torch.Generator(device="cpu").manual_seed(self.config["seed"])
        history: dict[str, list[float]] = {"epoch": [], "loss": []}

        for epoch in range(n_epochs):
            perm = torch.randperm(n, generator=gen).to(device_)
            x_b = x_full[perm[:batch_size]]
            y_b = y_full[perm[:batch_size]]
            z = self(x_b)
            loss = supcon_loss(z, y_b, temperature=temperature)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
            optim.step()
            history["epoch"].append(epoch)
            history["loss"].append(float(loss.item()))
            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                logger.info(
                    "CaseSignatureEncoder.fit: epoch %d/%d  supcon_loss=%.5f",
                    epoch + 1, n_epochs, history["loss"][-1],
                )

        self._is_fitted = True
        self._fit_history = history
        self.eval()
        return history

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> Path:
        """Save state_dict + config to a directory. Atomic per file."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # state_dict
        sd_path = path / "encoder.pt"
        sd_tmp = sd_path.with_suffix(".pt.tmp")
        torch.save(self.state_dict(), sd_tmp)
        os.replace(sd_tmp, sd_path)
        # config + fit metadata
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
            path / "encoder_config.json",
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
    ) -> "CaseSignatureEncoder":
        path = Path(path)
        meta_path = path / "encoder_config.json"
        sd_path = path / "encoder.pt"
        if not meta_path.is_file() or not sd_path.is_file():
            raise CaseMemoryError(
                f"CaseSignatureEncoder.load: missing encoder.pt or "
                f"encoder_config.json under {path}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _check_compatible_version(
            meta.get("schema_version", SCHEMA_VERSION),
            context=f"CaseSignatureEncoder@{path}",
        )
        encoder = cls(**meta["config"])
        encoder.load_state_dict(torch.load(sd_path, map_location=map_location))
        encoder._is_fitted = bool(meta.get("is_fitted", False))
        encoder.to(map_location)
        encoder.eval()
        return encoder


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def build_signature_matrix(
    library: CaseLibrary,
    encoder: CaseSignatureEncoder,
    *,
    case_ids: Optional[Sequence[str]] = None,
    device: Union[str, torch.device] = "cpu",
) -> tuple[tuple[str, ...], np.ndarray]:
    """Compute the signature matrix for retrieval.

    Parameters
    ----------
    library : CaseLibrary
    encoder : CaseSignatureEncoder
        Should be ``.fit``-ed; otherwise a warning is emitted.
    case_ids : sequence of str, optional
        Subset of case_ids to encode. Default: all cases, in sorted
        order.
    device : torch device for the encode pass.

    Returns
    -------
    (case_ids, signatures)
        ``case_ids`` is a tuple of length n_cases in the order the
        signatures appear in the matrix. ``signatures`` is a numpy
        array of shape ``(n_cases, signature_dim)``, rows L2-normalised.
    """
    if case_ids is None:
        case_ids = library.case_ids()
    if len(case_ids) == 0:
        return (), np.zeros((0, encoder.config["signature_dim"]), dtype=np.float32)
    trajectories = np.stack(
        [library[cid].pre_onset_trajectory for cid in case_ids], axis=0,
    ).astype(np.float32, copy=False)
    encoder.to(device)
    sig = encoder.encode(trajectories)
    return tuple(case_ids), sig


# ---------------------------------------------------------------------------
# CLI entry point: inspect / validate a library on disk
# ---------------------------------------------------------------------------


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> int:
    """CLI: ``python -m em_fin_stability.case_memory <command> [args]``.

    Commands
    --------
    inspect <library_path>
        Print a JSON summary of the library to stdout.
    validate <library_path>
        Re-load the library with strict checksum verification; exit
        code 0 on success, non-zero on any error.
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m em_fin_stability.case_memory",
        description="Inspect or validate a Crisis Case Library.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_inspect = sub.add_parser("inspect", help="Print library summary.")
    p_inspect.add_argument("library_path", type=Path)
    p_inspect.add_argument(
        "--no-verify", action="store_true",
        help="Skip checksum verification.",
    )
    p_validate = sub.add_parser("validate", help="Strict re-validation.")
    p_validate.add_argument("library_path", type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _configure_logging(args.log_level)

    try:
        if args.command == "inspect":
            lib = CaseLibrary.load(
                args.library_path,
                verify_checksums=not args.no_verify,
                strict=False,
            )
            print(json.dumps(lib.summary(), indent=2, sort_keys=True))
            return 0
        elif args.command == "validate":
            lib = CaseLibrary.load(
                args.library_path,
                verify_checksums=True,
                strict=True,
            )
            print(f"OK: {len(lib)} cases, checksum {lib.library_checksum()[:12]}")
            return 0
        else:  # pragma: no cover
            parser.print_help()
            return 2
    except CaseMemoryError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
