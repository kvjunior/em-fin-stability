"""
data_pipeline.py
================

Input-side data ingestion, alignment, and normalization for the
case-memory + analogy + mitigation pipeline.

This module is the bridge between raw observational data (per-country
quarterly macro series, crisis-date catalogues) and the trainable
modules of the framework: it produces normalised, aligned, fully-
provenanced ``case_memory.CrisisCase`` objects that feed
``DetectorEnsemble.fit`` and downstream training.

Architecture
------------
::

    RAW INPUTS
      ┌───────────────────────────────────┐
      │ per-country quarterly macro CSVs  │
      │ crisis-date catalogue CSV         │
      └─────────────┬─────────────────────┘
                    │
        ┌───────────┴──────────────┐
        ▼                          ▼
    MacroSeriesLoader      CrisisLabelLoader
        │                          │
        │  per-country DataFrame   │   crisis catalogue
        │                          │
        ▼                          │
    quality helpers                │
    (fill_short_gaps,              │
     winsorize,                    │
     compute_data_quality)         │
        │                          │
        ▼                          │
    MacroFeatureNormalizer         │
    (z-score with persisted stats) │
        │                          │
        └──────────┬───────────────┘
                   ▼
              CaseAssembler
        (aligns pre-onset/post-onset windows,
        attaches policy timeline, builds
        CrisisCase objects)
                   │
                   ▼
              DataPipeline
        (orchestrator: fit normaliser on
        training cases, then build a full
        CaseLibrary)
                   │
                   ▼
              CaseLibrary
        (consumed by case_memory, detection,
        coordinator, analogy_engine,
        mitigation)

Reproducibility commitments
---------------------------
* The normaliser persists per-feature mean and std to disk and
  re-applies them bit-identically at test time.
* All numeric helpers are pure functions with no global state.
* The synthetic library generator (used in test suites of the
  upstream modules) takes a seed and produces deterministic output.
* All disk writes are atomic.
* Provenance: every ``CrisisCase`` records the data sources that
  contributed (``MacroSeriesLoader@<path>``,
  ``CrisisLabelLoader@<path>``, normalizer fingerprint).

Why CSV inputs rather than direct API calls
-------------------------------------------
For academic reproducibility, raw data should be snapshotted, version-
controlled, and reusable. This module therefore expects CSV files on
disk rather than live API calls to IFS/WDI/IMF. Users who wish to
ingest fresh data should run a separate ingestion script that produces
the expected CSV schema; this pipeline then deterministically rebuilds
the case library from that snapshot.

Expected input schemas
----------------------
Per-country macro series CSV (one file per country, name = ``{ISO3}.csv``)::

    date,output_gap,inflation_yoy,reer_log_dev,...,output_gap_squared
    1995Q1,-0.012,8.34,-0.043,...,0.000144
    1995Q2,-0.008,7.91,-0.038,...,0.000064
    ...

* The ``date`` column uses ISO 8601 "YYYY-MM-DD" or pandas Period
  syntax "YYYYQn". Quarterly frequency.
* Feature columns must include all 12 ``DEFAULT_FEATURE_NAMES``, OR
  the derivable subset if ``feature_derivation=True``.

Crisis catalogue CSV::

    country_iso3,onset_year,onset_quarter,crisis_type,source,notes
    KOR,1997,4,currency,Laeven-Valencia 2018,
    KOR,1997,4,banking,Laeven-Valencia 2018,co-occurs with currency
    ...

* ``crisis_type`` must be one of ``case_memory.CRISIS_TYPES``.
* ``onset_quarter`` is a 1-indexed integer in {1, 2, 3, 4}.
* Composite types (``twin``, ``triple``) are listed as single rows.

References (APA-7)
------------------
Frankel, J., & Saravelos, G. (2012). Can leading indicators assess
    country vulnerability? Evidence from the 2008-09 global financial
    crisis. Journal of International Economics, 87(2), 216-231.

International Monetary Fund. (2024). International Financial
    Statistics (IFS) database. https://data.imf.org

Laeven, L., & Valencia, F. (2018). Systemic banking crises revisited
    (IMF Working Paper No. 18/206). International Monetary Fund.

Reinhart, C. M., & Rogoff, K. S. (2009). This time is different:
    Eight centuries of financial folly. Princeton University Press.

World Bank. (2024). World Development Indicators (WDI).
    https://databank.worldbank.org

Version
-------
1.0.0  Camera-ready KBS submission.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Union

import numpy as np

from case_memory import (
    AuthoritySnapshot,
    CRISIS_TYPES,
    CaseLibrary,
    CaseMemoryError,
    CrisisCase,
    DEFAULT_FEATURE_NAMES,
    DEFAULT_POST_ONSET_QUARTERS,
    DEFAULT_PRE_ONSET_QUARTERS,
    INSTITUTIONS,
    N_MACRO_FEATURES,
    PolicyAction,
)


__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "DEFAULT_MAX_GAP_QUARTERS",
    "DEFAULT_WINSORIZE_PCTILES",
    "DEFAULT_MIN_DATA_QUALITY",
    "DEFAULT_DATE_FORMATS",
    # Exceptions
    "DataPipelineError",
    "DataValidationError",
    "DataAlignmentError",
    "InsufficientHistoryError",
    "NormalizerNotFittedError",
    # Helpers
    "parse_date_to_quarter",
    "fill_short_gaps",
    "winsorize",
    "derive_features",
    "compute_data_quality",
    # Components
    "MacroFeatureNormalizer",
    "CrisisLabelLoader",
    "MacroSeriesLoader",
    "CaseAssembler",
    "DataPipeline",
    # Synthetic data
    "build_synthetic_library",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Semantic version of the on-disk data-pipeline artefacts.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: Maximum gap (in quarters) that may be linearly interpolated.
#: Gaps longer than this are left as NaN and cause a case to be
#: rejected by the assembler.
DEFAULT_MAX_GAP_QUARTERS: Final[int] = 2

#: Default winsorization percentiles. Caps each feature at the 1st
#: and 99th percentiles to handle EM hyperinflation episodes and
#: similar tail events.
DEFAULT_WINSORIZE_PCTILES: Final[tuple[float, float]] = (1.0, 99.0)

#: Cases with computed data_quality below this threshold are dropped
#: by the assembler with an explicit log message.
DEFAULT_MIN_DATA_QUALITY: Final[float] = 0.5

#: Accepted date formats for the ``date`` column of macro CSVs.
#: Parsed in order; first match wins.
DEFAULT_DATE_FORMATS: Final[tuple[str, ...]] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m",
    "%Y/%m",
)

#: Quarterly date pattern for inputs like "1995Q1" / "1995q1".
_QUARTERLY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<year>\d{4})[Qq](?P<q>[1-4])$"
)

#: Quarterly date pattern (separator) for "1995-Q1" / "1995_Q1".
_QUARTERLY_PATTERN_SEP: Final[re.Pattern[str]] = re.compile(
    r"^(?P<year>\d{4})[-_][Qq](?P<q>[1-4])$"
)

#: ISO3 country code pattern.
_ISO3_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{3}$")

#: Quarter span around onset: (pre, post). Default matches case_memory.
DEFAULT_PRE_QUARTERS: Final[int] = DEFAULT_PRE_ONSET_QUARTERS
DEFAULT_POST_QUARTERS: Final[int] = DEFAULT_POST_ONSET_QUARTERS

#: Tolerance for "approximately zero" sample variance in the
#: normaliser. Below this, std is treated as 1.0 to avoid divide-by-zero.
_STD_FLOOR: Final[float] = 1e-8


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DataPipelineError(CaseMemoryError):
    """Base class for data-pipeline exceptions."""


class DataValidationError(DataPipelineError):
    """Input data violated a schema or value constraint."""


class DataAlignmentError(DataPipelineError):
    """Pre/post-onset window could not be aligned around the onset
    quarter (e.g., insufficient history before the onset)."""


class InsufficientHistoryError(DataAlignmentError):
    """A specific subclass of DataAlignmentError when there are
    fewer than ``T_pre`` quarters of macro history before onset."""


class NormalizerNotFittedError(DataPipelineError):
    """``MacroFeatureNormalizer.transform`` was called before ``fit``."""


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
        raise DataPipelineError(
            f"{context}: malformed schema_version {found_version!r}; "
            f"expected MAJOR.MINOR.PATCH"
        ) from exc
    if len(found) != 3 or len(expected) != 3:
        raise DataPipelineError(
            f"{context}: schema_version must be MAJOR.MINOR.PATCH; "
            f"got {found_version!r}"
        )
    if found[0] != expected[0]:
        raise DataPipelineError(
            f"{context}: schema major-version mismatch (found "
            f"{found_version!r}, code supports {expected_version!r})"
        )
    if found[1] != expected[1]:
        logger.warning(
            "%s: schema minor-version mismatch (found %s, code supports %s)",
            context, found_version, expected_version,
        )


# ---------------------------------------------------------------------------
# Public helpers: date parsing and quality control
# ---------------------------------------------------------------------------


def parse_date_to_quarter(s: Union[str, datetime]) -> tuple[int, int]:
    """Parse a date string into ``(year, quarter)`` where quarter is 1-4.

    Accepted formats:
      * "1995Q1", "1995q1" (quarterly literal, no separator)
      * "1995-Q1", "1995_Q1" (quarterly literal with separator)
      * "1995-01-01", "1995/01/01" (ISO 8601 day; quarter inferred from month)
      * "1995-01", "1995/01" (year-month; quarter inferred from month)
      * ``datetime`` object (quarter inferred from .month)

    Parameters
    ----------
    s : str or datetime

    Returns
    -------
    (year, quarter) : tuple[int, int]
    """
    if isinstance(s, datetime):
        return int(s.year), int((s.month - 1) // 3 + 1)
    if not isinstance(s, str):
        raise DataValidationError(
            f"parse_date_to_quarter: must be str or datetime; got {type(s)}"
        )
    s = s.strip()
    if not s:
        raise DataValidationError("parse_date_to_quarter: empty string")
    # Try the quarterly patterns first
    for pat in (_QUARTERLY_PATTERN, _QUARTERLY_PATTERN_SEP):
        m = pat.match(s)
        if m:
            year = int(m.group("year"))
            q = int(m.group("q"))
            return year, q
    # Then the day/month formats
    for fmt in DEFAULT_DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            return int(dt.year), int((dt.month - 1) // 3 + 1)
        except ValueError:
            continue
    raise DataValidationError(
        f"parse_date_to_quarter: could not parse {s!r}; "
        f"expected one of {list(DEFAULT_DATE_FORMATS) + ['YYYYQn']}"
    )


def _quarter_index(year: int, quarter: int, *, base_year: int) -> int:
    """Linear index in quarters from ``(base_year, Q1)``."""
    if quarter < 1 or quarter > 4:
        raise DataValidationError(
            f"_quarter_index: quarter must be in [1, 4]; got {quarter}"
        )
    return (year - base_year) * 4 + (quarter - 1)


def fill_short_gaps(
    arr: np.ndarray,
    *,
    max_gap_quarters: int = DEFAULT_MAX_GAP_QUARTERS,
    return_mask: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Linearly interpolate NaN gaps up to ``max_gap_quarters`` long.

    Operates column-wise on a 2-D array of shape ``(T, F)``. Gaps
    longer than ``max_gap_quarters`` are left as NaN; the caller is
    expected to either reject such cases or handle them explicitly.

    Edge gaps (NaN runs at the start or end of a column) are also
    capped at ``max_gap_quarters`` — leading NaNs are back-filled
    from the first observed value, trailing NaNs are forward-filled
    from the last observed value.

    Parameters
    ----------
    arr : np.ndarray of shape ``(T, F)`` or ``(T,)``
        Quarterly time series. NaN marks missing values.
    max_gap_quarters : int
        Maximum gap length (in quarters) that may be filled.
        Set to 0 to disable interpolation; set to a very large
        number to fill all internal gaps regardless of length.
    return_mask : bool
        If True, also return a boolean mask of the same shape
        marking originally-NaN positions that were filled
        (``True`` = was NaN and is now filled).

    Returns
    -------
    filled : np.ndarray
        Same shape as input, with short gaps linearly interpolated.
    mask : np.ndarray of bool (only if ``return_mask=True``)
        Positions that were originally NaN and have now been filled.
    """
    if not isinstance(arr, np.ndarray):
        raise DataValidationError(
            f"fill_short_gaps: arr must be np.ndarray; got {type(arr)}"
        )
    if max_gap_quarters < 0:
        raise DataValidationError(
            f"max_gap_quarters must be >= 0; got {max_gap_quarters}"
        )
    is_1d = (arr.ndim == 1)
    if is_1d:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise DataValidationError(
            f"fill_short_gaps: arr must be 1-D or 2-D; got shape {arr.shape}"
        )
    out = arr.astype(np.float64, copy=True)
    T, F = out.shape
    mask_filled = np.zeros_like(out, dtype=bool)

    for f in range(F):
        col = out[:, f]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue
        # Find runs of NaN
        # (start, length) pairs
        i = 0
        runs: list[tuple[int, int]] = []
        while i < T:
            if not nan_mask[i]:
                i += 1
                continue
            j = i
            while j < T and nan_mask[j]:
                j += 1
            runs.append((i, j - i))
            i = j

        for start, length in runs:
            if length > max_gap_quarters:
                continue
            end = start + length  # exclusive
            left_idx = start - 1
            right_idx = end
            if left_idx >= 0 and right_idx < T:
                # Interior gap
                left = col[left_idx]
                right = col[right_idx]
                for k in range(length):
                    t = (k + 1) / (length + 1)
                    col[start + k] = left + t * (right - left)
                    mask_filled[start + k, f] = True
            elif left_idx < 0 and right_idx < T:
                # Leading edge gap: back-fill from first observation
                fill_val = col[right_idx]
                for k in range(length):
                    col[start + k] = fill_val
                    mask_filled[start + k, f] = True
            elif left_idx >= 0 and right_idx >= T:
                # Trailing edge gap: forward-fill from last observation
                fill_val = col[left_idx]
                for k in range(length):
                    col[start + k] = fill_val
                    mask_filled[start + k, f] = True
            # else: column is entirely NaN; leave as-is
        out[:, f] = col

    if is_1d:
        out = out[:, 0]
        mask_filled = mask_filled[:, 0]
    if return_mask:
        return out, mask_filled
    return out


def winsorize(
    arr: np.ndarray,
    *,
    pctiles: tuple[float, float] = DEFAULT_WINSORIZE_PCTILES,
    return_mask: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Cap values at the (low, high) percentiles of each column.

    Operates column-wise on a 2-D array. Used to handle EM
    hyperinflation episodes and similar tail events that distort
    z-score normalisation.

    Parameters
    ----------
    arr : np.ndarray of shape ``(T, F)`` or ``(T,)``
    pctiles : (low, high)
        Percentiles in [0, 100]. Defaults to (1.0, 99.0).
    return_mask : bool
        If True, also return a boolean mask marking positions that
        were capped (``True`` = was outside [low, high] before).

    Returns
    -------
    capped : np.ndarray
    mask : np.ndarray of bool (only if ``return_mask=True``)
    """
    low_p, high_p = pctiles
    if not 0 <= low_p < high_p <= 100:
        raise DataValidationError(
            f"winsorize: pctiles must satisfy 0 <= low < high <= 100; "
            f"got {pctiles}"
        )
    if not isinstance(arr, np.ndarray):
        raise DataValidationError(
            f"winsorize: arr must be np.ndarray; got {type(arr)}"
        )
    is_1d = (arr.ndim == 1)
    if is_1d:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise DataValidationError(
            f"winsorize: arr must be 1-D or 2-D; got shape {arr.shape}"
        )
    out = arr.astype(np.float64, copy=True)
    T, F = out.shape
    mask_capped = np.zeros_like(out, dtype=bool)
    for f in range(F):
        col = out[:, f]
        finite_mask = np.isfinite(col)
        if not finite_mask.any():
            continue
        finite_vals = col[finite_mask]
        low = float(np.percentile(finite_vals, low_p))
        high = float(np.percentile(finite_vals, high_p))
        if not np.isfinite(low) or not np.isfinite(high):
            continue
        cap_low = (col < low) & finite_mask
        cap_high = (col > high) & finite_mask
        mask_capped[cap_low, f] = True
        mask_capped[cap_high, f] = True
        col[cap_low] = low
        col[cap_high] = high
        out[:, f] = col
    if is_1d:
        out = out[:, 0]
        mask_capped = mask_capped[:, 0]
    if return_mask:
        return out, mask_capped
    return out


def derive_features(
    df: dict[str, np.ndarray],
    *,
    features: Sequence[str] = DEFAULT_FEATURE_NAMES,
) -> dict[str, np.ndarray]:
    """Compute derived features that are common-knowledge transformations
    of more primitive ones.

    Currently supported:
      * ``inflation_yoy_change`` <- first-difference of ``inflation_yoy``
      * ``output_gap_squared`` <- elementwise square of ``output_gap``

    Parameters
    ----------
    df : dict[str, np.ndarray]
        Per-column 1-D arrays sharing the same time axis. May already
        contain the derived columns (in which case they are not
        recomputed).
    features : Sequence[str]
        The target feature names. Only derivable names in ``features``
        will be computed.

    Returns
    -------
    df_out : dict[str, np.ndarray]
        Shallow-copy of input with derived columns added where possible.
    """
    df_out = dict(df)  # shallow copy
    if "inflation_yoy_change" in features and \
            "inflation_yoy_change" not in df_out and \
            "inflation_yoy" in df_out:
        src = df_out["inflation_yoy"]
        diff = np.empty_like(src, dtype=np.float64)
        diff[0] = 0.0
        diff[1:] = src[1:] - src[:-1]
        df_out["inflation_yoy_change"] = diff
    if "output_gap_squared" in features and \
            "output_gap_squared" not in df_out and \
            "output_gap" in df_out:
        df_out["output_gap_squared"] = df_out["output_gap"] ** 2
    return df_out


def compute_data_quality(
    n_observed: int,
    n_filled: int,
    n_winsorized: int,
    n_total: int,
) -> float:
    """Map per-case data-completeness statistics to a quality score
    in ``[0, 1]``.

    The score weights filled values at 0.5 weight (a filled value is
    less reliable than an observed one but more reliable than no data)
    and winsorized values at 0.8 weight (a winsorized value retains
    most of its information; only the magnitude is clipped).
    Unobserved-and-unfilled values count as zero.

    Formula:
      ``quality = (n_observed + 0.5 * n_filled + (-0.2) * n_winsorized)
                  / n_total``

    Note that ``n_winsorized`` is subtracted with a 0.2 weight (so a
    winsorized value contributes 0.8 to the numerator if it was
    originally observed, since it adds 1.0 then subtracts 0.2).
    Likewise a filled-and-winsorized value contributes 0.3.

    Parameters
    ----------
    n_observed : int
        Number of cells that were originally observed (not NaN).
    n_filled : int
        Number of cells that were originally NaN and have been
        interpolated.
    n_winsorized : int
        Number of cells whose value was capped at a percentile.
    n_total : int
        Total number of cells in the trajectory (T * F).

    Returns
    -------
    quality : float in [0, 1]
    """
    if n_total <= 0:
        raise DataValidationError("compute_data_quality: n_total must be > 0")
    if n_observed < 0 or n_filled < 0 or n_winsorized < 0:
        raise DataValidationError(
            "compute_data_quality: counts must be >= 0"
        )
    if n_observed + n_filled > n_total:
        raise DataValidationError(
            "compute_data_quality: observed+filled exceeds total"
        )
    q = (n_observed + 0.5 * n_filled - 0.2 * n_winsorized) / n_total
    return float(np.clip(q, 0.0, 1.0))


# ---------------------------------------------------------------------------
# MacroFeatureNormalizer
# ---------------------------------------------------------------------------


class MacroFeatureNormalizer:
    """Z-score normalization with persisted per-feature statistics.

    This is the most critical component for reproducibility: the
    statistics fit on the training cases must be re-applied bit-
    identically when the pipeline is used on new test cases (or at
    inference time). A fresh ``.fit`` call on test data would invalidate
    the training distribution and ruin any downstream model.

    Conventions
    -----------
    * Per-feature mean and std are stored in two arrays of shape ``(F,)``.
    * For features where the sample std is below ``_STD_FLOOR``, std is
      treated as 1.0; this only happens for constant features which
      shouldn't appear in practice.
    * Saved to disk as a JSON manifest with deterministic key order.

    Parameters
    ----------
    feature_names : tuple[str, ...]
        Ordered feature names that this normaliser knows about.
    """

    def __init__(
        self, *,
        feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
    ) -> None:
        if len(feature_names) == 0:
            raise DataValidationError(
                "MacroFeatureNormalizer: feature_names must not be empty"
            )
        if len(set(feature_names)) != len(feature_names):
            raise DataValidationError(
                "MacroFeatureNormalizer: feature_names must be unique"
            )
        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._n_samples_fit: int = 0
        self._fit_at: str = ""

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None and self._std is not None

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def mean(self) -> np.ndarray:
        if self._mean is None:
            raise NormalizerNotFittedError(
                "MacroFeatureNormalizer.mean: not fitted yet"
            )
        return self._mean.copy()

    @property
    def std(self) -> np.ndarray:
        if self._std is None:
            raise NormalizerNotFittedError(
                "MacroFeatureNormalizer.std: not fitted yet"
            )
        return self._std.copy()

    def fit(self, X: np.ndarray) -> "MacroFeatureNormalizer":
        """Fit per-feature mean and std from a 2-D or 3-D array.

        Parameters
        ----------
        X : np.ndarray
            * 2-D shape ``(N, F)``: each row is one observation
            * 3-D shape ``(N, T, F)``: each ``(T, F)`` block is one
              trajectory; statistics are aggregated across all
              ``N * T`` rows.

        Returns
        -------
        self : MacroFeatureNormalizer
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:
            X2 = X.reshape(-1, X.shape[-1])
        elif X.ndim == 2:
            X2 = X
        else:
            raise DataValidationError(
                f"MacroFeatureNormalizer.fit: X must be 2-D or 3-D; "
                f"got shape {X.shape}"
            )
        if X2.shape[1] != self.n_features:
            raise DataValidationError(
                f"MacroFeatureNormalizer.fit: X has {X2.shape[1]} features, "
                f"but normalizer was created with {self.n_features}"
            )
        if X2.shape[0] < 2:
            raise DataValidationError(
                f"MacroFeatureNormalizer.fit: need >= 2 samples; "
                f"got {X2.shape[0]}"
            )
        if not np.all(np.isfinite(X2)):
            raise DataValidationError(
                "MacroFeatureNormalizer.fit: X contains non-finite entries"
            )
        self._mean = X2.mean(axis=0).astype(np.float64)
        std = X2.std(axis=0, ddof=1).astype(np.float64)  # sample std
        # Floor near-zero std to avoid divide-by-zero
        self._std = np.where(std < _STD_FLOOR, 1.0, std)
        self._n_samples_fit = int(X2.shape[0])
        self._fit_at = _utc_now_iso()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to a 2-D or 3-D array.

        Parameters
        ----------
        X : np.ndarray of shape ``(..., F)``
            Any leading dimensions are preserved.

        Returns
        -------
        Z : np.ndarray of same shape as X.
        """
        if not self.is_fitted:
            raise NormalizerNotFittedError(
                "MacroFeatureNormalizer.transform: not fitted yet"
            )
        X = np.asarray(X, dtype=np.float64)
        if X.shape[-1] != self.n_features:
            raise DataValidationError(
                f"MacroFeatureNormalizer.transform: last axis "
                f"{X.shape[-1]} != n_features {self.n_features}"
            )
        return (X - self._mean) / self._std

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map normalized values back to the original scale."""
        if not self.is_fitted:
            raise NormalizerNotFittedError(
                "MacroFeatureNormalizer.inverse_transform: not fitted yet"
            )
        Z = np.asarray(Z, dtype=np.float64)
        if Z.shape[-1] != self.n_features:
            raise DataValidationError(
                f"MacroFeatureNormalizer.inverse_transform: last axis "
                f"{Z.shape[-1]} != n_features {self.n_features}"
            )
        return Z * self._std + self._mean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: ``fit(X).transform(X)``."""
        self.fit(X)
        return self.transform(X)

    def fingerprint(self) -> str:
        """SHA-256 hex digest of the fitted statistics.

        Used by ``CrisisCase.provenance`` to record which normalizer
        was applied during case construction. Two normalisers with
        the same mean/std arrays produce the same fingerprint.
        """
        if not self.is_fitted:
            return "unfitted"
        payload = {
            "feature_names": list(self.feature_names),
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
        }
        return _sha256_hex(_canonical_json(payload))[:16]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "feature_names": list(self.feature_names),
            "is_fitted": self.is_fitted,
            "n_samples_fit": int(self._n_samples_fit),
            "fit_at": self._fit_at,
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
            "fingerprint": self.fingerprint(),
            "saved_at": _utc_now_iso(),
        }
        _atomic_write_text(
            path,
            json.dumps(manifest, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "MacroFeatureNormalizer":
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _check_compatible_version(
            manifest.get("schema_version", SCHEMA_VERSION),
            context=f"MacroFeatureNormalizer@{path}",
        )
        norm = cls(feature_names=tuple(manifest["feature_names"]))
        if manifest.get("is_fitted") and manifest.get("mean") is not None:
            norm._mean = np.asarray(manifest["mean"], dtype=np.float64)
            norm._std = np.asarray(manifest["std"], dtype=np.float64)
            norm._n_samples_fit = int(manifest.get("n_samples_fit", 0))
            norm._fit_at = str(manifest.get("fit_at", ""))
            # Round-trip check
            expected_fp = manifest.get("fingerprint")
            if expected_fp is not None and norm.fingerprint() != expected_fp:
                raise DataPipelineError(
                    f"MacroFeatureNormalizer.load@{path}: fingerprint "
                    f"mismatch (expected {expected_fp}, got "
                    f"{norm.fingerprint()})"
                )
        return norm

    def __repr__(self) -> str:
        return (
            f"MacroFeatureNormalizer(F={self.n_features}, "
            f"fitted={self.is_fitted}, fingerprint={self.fingerprint()})"
        )


# ---------------------------------------------------------------------------
# CrisisLabelLoader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrisisLabel:
    """One row of a crisis catalogue."""
    country_iso3: str
    onset_year: int
    onset_quarter: int
    crisis_type: str
    source: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not _ISO3_PATTERN.match(self.country_iso3):
            raise DataValidationError(
                f"CrisisLabel: country_iso3 {self.country_iso3!r} must be "
                f"a 3-letter uppercase code"
            )
        if not 1800 <= self.onset_year <= 2100:
            raise DataValidationError(
                f"CrisisLabel: onset_year {self.onset_year} out of range"
            )
        if self.onset_quarter not in (1, 2, 3, 4):
            raise DataValidationError(
                f"CrisisLabel: onset_quarter {self.onset_quarter} not in "
                f"[1, 4]"
            )
        if self.crisis_type not in CRISIS_TYPES:
            raise DataValidationError(
                f"CrisisLabel: crisis_type {self.crisis_type!r} not in "
                f"{list(CRISIS_TYPES)}"
            )

    def case_id(self) -> str:
        return f"{self.country_iso3}_{self.onset_year}_Q{self.onset_quarter}"


class CrisisLabelLoader:
    """Reads a crisis catalogue from a CSV file.

    Expected columns (case-sensitive):
      ``country_iso3``, ``onset_year``, ``onset_quarter``,
      ``crisis_type``, ``source``, ``notes``

    The CSV is parsed using only the standard library — no pandas
    dependency — so the loader is robust to environment differences.

    Duplicate rows (same ``(country, year, quarter, crisis_type)``)
    are deduplicated, with a warning logged.
    """

    def __init__(self, csv_path: Path) -> None:
        self.csv_path: Path = Path(csv_path)
        if not self.csv_path.is_file():
            raise DataPipelineError(
                f"CrisisLabelLoader: file not found: {self.csv_path}"
            )

    def load(self) -> list[CrisisLabel]:
        """Parse and validate the catalogue CSV.

        Returns
        -------
        labels : list[CrisisLabel]
        """
        import csv
        rows_seen: set[tuple[str, int, int, str]] = set()
        out: list[CrisisLabel] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"country_iso3", "onset_year", "onset_quarter",
                        "crisis_type", "source"}
            if reader.fieldnames is None:
                raise DataValidationError(
                    f"CrisisLabelLoader: CSV {self.csv_path} has no header"
                )
            missing = required - set(reader.fieldnames)
            if missing:
                raise DataValidationError(
                    f"CrisisLabelLoader: CSV {self.csv_path} missing "
                    f"required columns {sorted(missing)}"
                )
            for line_no, row in enumerate(reader, start=2):
                try:
                    label = CrisisLabel(
                        country_iso3=row["country_iso3"].strip().upper(),
                        onset_year=int(row["onset_year"]),
                        onset_quarter=int(row["onset_quarter"]),
                        crisis_type=row["crisis_type"].strip().lower(),
                        source=row["source"].strip(),
                        notes=(row.get("notes") or "").strip(),
                    )
                except (ValueError, DataValidationError) as exc:
                    raise DataValidationError(
                        f"CrisisLabelLoader: {self.csv_path}:{line_no}: "
                        f"{exc}"
                    ) from exc
                key = (label.country_iso3, label.onset_year,
                       label.onset_quarter, label.crisis_type)
                if key in rows_seen:
                    logger.warning(
                        "CrisisLabelLoader: duplicate row at %s:%d "
                        "(country=%s year=%d Q%d type=%s); ignoring",
                        self.csv_path, line_no, label.country_iso3,
                        label.onset_year, label.onset_quarter,
                        label.crisis_type,
                    )
                    continue
                rows_seen.add(key)
                out.append(label)
        return out

    def __repr__(self) -> str:
        return f"CrisisLabelLoader(csv_path={self.csv_path})"


# ---------------------------------------------------------------------------
# MacroSeriesLoader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroSeries:
    """One country's quarterly macro series."""
    country_iso3: str
    years: np.ndarray  # int64, shape (T,)
    quarters: np.ndarray  # int64, shape (T,)
    data: np.ndarray  # float64, shape (T, F)
    feature_names: tuple[str, ...]
    source_path: Path = field(default_factory=lambda: Path("<memory>"))

    def __post_init__(self) -> None:
        T = self.data.shape[0]
        if self.data.ndim != 2:
            raise DataValidationError(
                f"MacroSeries: data must be 2-D; got shape {self.data.shape}"
            )
        if self.years.shape != (T,):
            raise DataValidationError(
                f"MacroSeries: years shape {self.years.shape} != ({T},)"
            )
        if self.quarters.shape != (T,):
            raise DataValidationError(
                f"MacroSeries: quarters shape {self.quarters.shape} "
                f"!= ({T},)"
            )
        if self.data.shape[1] != len(self.feature_names):
            raise DataValidationError(
                f"MacroSeries: data has {self.data.shape[1]} columns but "
                f"{len(self.feature_names)} feature_names"
            )
        # Quarters strictly increasing
        if T > 1:
            qidx = self.years.astype(np.int64) * 4 + (self.quarters - 1)
            if not np.all(np.diff(qidx) > 0):
                raise DataValidationError(
                    f"MacroSeries[{self.country_iso3}]: dates not strictly "
                    f"increasing"
                )

    @property
    def n_quarters(self) -> int:
        return int(self.data.shape[0])

    def find_quarter_index(
        self, year: int, quarter: int,
    ) -> Optional[int]:
        """Linear index into ``data`` of the row matching ``(year, quarter)``,
        or ``None`` if not present."""
        match = (self.years == year) & (self.quarters == quarter)
        idx = np.where(match)[0]
        if len(idx) == 0:
            return None
        return int(idx[0])


class MacroSeriesLoader:
    """Loads per-country quarterly macro CSVs from a directory.

    File naming convention: ``{ISO3}.csv`` (e.g., ``KOR.csv``).
    Each file must have a ``date`` column and one column per
    feature in ``feature_names``. Other columns are ignored.

    Features that are missing from the input CSV but can be derived
    (``inflation_yoy_change``, ``output_gap_squared``) are computed
    via ``derive_features``. All other missing features raise.
    """

    def __init__(
        self,
        directory: Path,
        *,
        feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
        max_gap_quarters: int = DEFAULT_MAX_GAP_QUARTERS,
        winsorize_pctiles: Optional[tuple[float, float]] = DEFAULT_WINSORIZE_PCTILES,
        derive: bool = True,
    ) -> None:
        self.directory: Path = Path(directory)
        if not self.directory.is_dir():
            raise DataPipelineError(
                f"MacroSeriesLoader: directory not found: {self.directory}"
            )
        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self.max_gap_quarters: int = int(max_gap_quarters)
        self.winsorize_pctiles: Optional[tuple[float, float]] = (
            tuple(winsorize_pctiles) if winsorize_pctiles else None
        )
        self.derive: bool = bool(derive)

    def load_country(self, country_iso3: str) -> MacroSeries:
        """Load one country's series.

        Raises
        ------
        DataPipelineError
            If the file is missing.
        DataValidationError
            If the file's schema is malformed.
        """
        import csv
        country_iso3 = country_iso3.strip().upper()
        if not _ISO3_PATTERN.match(country_iso3):
            raise DataValidationError(
                f"MacroSeriesLoader: invalid ISO3 {country_iso3!r}"
            )
        csv_path = self.directory / f"{country_iso3}.csv"
        if not csv_path.is_file():
            raise DataPipelineError(
                f"MacroSeriesLoader: file not found: {csv_path}"
            )

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise DataValidationError(
                    f"MacroSeriesLoader: CSV {csv_path} has no header"
                )
            if "date" not in reader.fieldnames:
                raise DataValidationError(
                    f"MacroSeriesLoader: CSV {csv_path} missing 'date' column"
                )
            present_features = set(reader.fieldnames) - {"date"}

            years_list: list[int] = []
            quarters_list: list[int] = []
            per_feature_lists: dict[str, list[float]] = {
                name: [] for name in present_features
            }

            for line_no, row in enumerate(reader, start=2):
                try:
                    y, q = parse_date_to_quarter(row["date"])
                except DataValidationError as exc:
                    raise DataValidationError(
                        f"MacroSeriesLoader: {csv_path}:{line_no}: "
                        f"bad date: {exc}"
                    ) from exc
                years_list.append(y)
                quarters_list.append(q)
                for name in present_features:
                    s = (row.get(name) or "").strip()
                    if s == "" or s.lower() in {"nan", "na", "null"}:
                        per_feature_lists[name].append(float("nan"))
                    else:
                        try:
                            per_feature_lists[name].append(float(s))
                        except ValueError as exc:
                            raise DataValidationError(
                                f"MacroSeriesLoader: {csv_path}:{line_no}: "
                                f"column {name!r}: not a float ({s!r})"
                            ) from exc

        # Apply derivations if enabled
        per_feature_arr: dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in per_feature_lists.items()
        }
        if self.derive:
            per_feature_arr = derive_features(
                per_feature_arr, features=self.feature_names,
            )

        # Validate that all required features are present
        missing = set(self.feature_names) - set(per_feature_arr.keys())
        if missing:
            raise DataValidationError(
                f"MacroSeriesLoader[{country_iso3}]: missing required "
                f"features {sorted(missing)} (after derivation). "
                f"Present: {sorted(per_feature_arr.keys())}"
            )

        T = len(years_list)
        if T == 0:
            raise DataValidationError(
                f"MacroSeriesLoader[{country_iso3}]: CSV has no data rows"
            )
        data = np.zeros((T, len(self.feature_names)), dtype=np.float64)
        for j, name in enumerate(self.feature_names):
            data[:, j] = per_feature_arr[name]

        # Sort by date in case the CSV was unordered
        years_arr = np.asarray(years_list, dtype=np.int64)
        quarters_arr = np.asarray(quarters_list, dtype=np.int64)
        order = np.lexsort((quarters_arr, years_arr))
        years_arr = years_arr[order]
        quarters_arr = quarters_arr[order]
        data = data[order]

        # Fill short gaps then winsorize
        if self.max_gap_quarters > 0:
            data = fill_short_gaps(
                data, max_gap_quarters=self.max_gap_quarters,
            )
        if self.winsorize_pctiles is not None:
            data = winsorize(data, pctiles=self.winsorize_pctiles)

        return MacroSeries(
            country_iso3=country_iso3,
            years=years_arr, quarters=quarters_arr,
            data=data, feature_names=self.feature_names,
            source_path=csv_path,
        )

    def list_available_countries(self) -> list[str]:
        """List ISO3 codes for which a CSV file exists in the directory."""
        out = []
        for p in self.directory.glob("*.csv"):
            stem = p.stem.upper()
            if _ISO3_PATTERN.match(stem):
                out.append(stem)
        return sorted(out)

    def __repr__(self) -> str:
        return (
            f"MacroSeriesLoader(directory={self.directory}, "
            f"F={len(self.feature_names)})"
        )


# ---------------------------------------------------------------------------
# CaseAssembler
# ---------------------------------------------------------------------------


class CaseAssembler:
    """Builds ``CrisisCase`` objects by aligning macro series around
    crisis-onset quarters.

    The assembler extracts a pre-onset window of ``pre_quarters``
    quarters ending at quarter ``onset - 1`` (inclusive of onset-1,
    exclusive of onset) and a post-onset window of ``post_quarters``
    quarters starting at quarter ``onset`` (inclusive). Both windows
    are taken from the country's ``MacroSeries`` and (optionally)
    normalised by an upstream ``MacroFeatureNormalizer``.

    Each constructed ``CrisisCase`` carries provenance recording the
    source CSV files and the normaliser fingerprint, and a
    ``data_quality`` score computed from the fill/winsorize history.
    """

    def __init__(
        self, *,
        pre_quarters: int = DEFAULT_PRE_QUARTERS,
        post_quarters: int = DEFAULT_POST_QUARTERS,
        normalizer: Optional[MacroFeatureNormalizer] = None,
        feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
        min_data_quality: float = DEFAULT_MIN_DATA_QUALITY,
        labels_loader_path: Optional[Path] = None,
        series_loader_dir: Optional[Path] = None,
    ) -> None:
        if pre_quarters < 1:
            raise DataValidationError(
                f"pre_quarters must be >= 1; got {pre_quarters}"
            )
        if post_quarters < 1:
            raise DataValidationError(
                f"post_quarters must be >= 1; got {post_quarters}"
            )
        if not 0.0 <= min_data_quality <= 1.0:
            raise DataValidationError(
                f"min_data_quality must be in [0, 1]; got {min_data_quality}"
            )
        if normalizer is not None and \
                normalizer.feature_names != tuple(feature_names):
            raise DataValidationError(
                "CaseAssembler: normalizer.feature_names "
                f"{normalizer.feature_names} != feature_names "
                f"{tuple(feature_names)}"
            )
        self.pre_quarters: int = int(pre_quarters)
        self.post_quarters: int = int(post_quarters)
        self.normalizer: Optional[MacroFeatureNormalizer] = normalizer
        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self.min_data_quality: float = float(min_data_quality)
        self._labels_loader_path = (
            Path(labels_loader_path) if labels_loader_path else None
        )
        self._series_loader_dir = (
            Path(series_loader_dir) if series_loader_dir else None
        )

    def assemble_one(
        self,
        label: CrisisLabel,
        series: MacroSeries,
        *,
        raw_data_for_quality: Optional[np.ndarray] = None,
    ) -> CrisisCase:
        """Build a single ``CrisisCase`` from one label and one series.

        Parameters
        ----------
        label : CrisisLabel
        series : MacroSeries
            Must be for the same country as ``label.country_iso3``.
        raw_data_for_quality : np.ndarray, optional
            Pre-fill/winsorize raw trajectory (same shape as the
            assembled pre+post window). Used to compute
            ``data_quality``. If not provided, quality defaults to
            1.0.

        Returns
        -------
        case : CrisisCase

        Raises
        ------
        DataAlignmentError, InsufficientHistoryError
            If alignment cannot be performed.
        """
        if label.country_iso3 != series.country_iso3:
            raise DataAlignmentError(
                f"CaseAssembler: label country {label.country_iso3} != "
                f"series country {series.country_iso3}"
            )
        if series.feature_names != self.feature_names:
            raise DataValidationError(
                f"CaseAssembler: series.feature_names "
                f"{series.feature_names} != expected {self.feature_names}"
            )
        # Locate onset
        onset_idx = series.find_quarter_index(
            label.onset_year, label.onset_quarter,
        )
        if onset_idx is None:
            raise DataAlignmentError(
                f"CaseAssembler[{label.case_id()}]: onset quarter "
                f"{label.onset_year}Q{label.onset_quarter} not present in "
                f"series (range {int(series.years[0])}Q"
                f"{int(series.quarters[0])} .. {int(series.years[-1])}Q"
                f"{int(series.quarters[-1])})"
            )
        # Pre-window: quarters [onset_idx - pre_quarters .. onset_idx) (exclusive of onset)
        pre_start = onset_idx - self.pre_quarters
        if pre_start < 0:
            raise InsufficientHistoryError(
                f"CaseAssembler[{label.case_id()}]: need {self.pre_quarters} "
                f"quarters before onset, but only {onset_idx} available"
            )
        # Post-window: quarters [onset_idx .. onset_idx + post_quarters)
        post_end = onset_idx + self.post_quarters
        if post_end > series.n_quarters:
            raise InsufficientHistoryError(
                f"CaseAssembler[{label.case_id()}]: need {self.post_quarters} "
                f"quarters after onset, but only "
                f"{series.n_quarters - onset_idx} available"
            )
        pre = series.data[pre_start:onset_idx].copy()  # (T_pre, F)
        post = series.data[onset_idx:post_end].copy()  # (T_post, F)

        # Sanity: no NaN in the assembled windows (would mean gap > max_gap_quarters)
        if not np.all(np.isfinite(pre)):
            raise DataAlignmentError(
                f"CaseAssembler[{label.case_id()}]: pre-onset window "
                f"contains non-finite values (gaps too long for "
                f"max_gap_quarters)"
            )
        if not np.all(np.isfinite(post)):
            raise DataAlignmentError(
                f"CaseAssembler[{label.case_id()}]: post-onset window "
                f"contains non-finite values"
            )

        # Normalise if requested
        if self.normalizer is not None:
            pre = self.normalizer.transform(pre)
            post = self.normalizer.transform(post)

        # Data quality
        if raw_data_for_quality is not None:
            raw = np.asarray(raw_data_for_quality, dtype=np.float64)
            expected_shape = (
                self.pre_quarters + self.post_quarters,
                len(self.feature_names),
            )
            if raw.shape != expected_shape:
                raise DataValidationError(
                    f"CaseAssembler[{label.case_id()}]: raw_data_for_quality "
                    f"shape {raw.shape} != expected {expected_shape}"
                )
            n_total = int(raw.size)
            nan_mask = np.isnan(raw)
            n_observed = int(np.sum(~nan_mask))
            n_filled = int(np.sum(nan_mask))  # filled iff was NaN and is now present in series
            dq = compute_data_quality(
                n_observed=n_observed,
                n_filled=n_filled,
                n_winsorized=0,  # winsorization tracked elsewhere if needed
                n_total=n_total,
            )
        else:
            dq = 1.0

        # Build provenance tuple
        provenance_parts: list[str] = []
        if self._labels_loader_path is not None:
            provenance_parts.append(
                f"CrisisLabelLoader@{self._labels_loader_path}"
            )
        else:
            provenance_parts.append(f"CrisisLabel:{label.source}")
        if self._series_loader_dir is not None:
            provenance_parts.append(
                f"MacroSeriesLoader@{self._series_loader_dir}"
            )
        if self.normalizer is not None:
            provenance_parts.append(
                f"MacroFeatureNormalizer:{self.normalizer.fingerprint()}"
            )

        return CrisisCase(
            case_id=label.case_id(),
            country_iso3=label.country_iso3,
            onset_year=label.onset_year,
            onset_quarter=label.onset_quarter,
            crisis_type=label.crisis_type,
            pre_onset_trajectory=pre,
            post_onset_trajectory=post,
            feature_names=self.feature_names,
            policy_timeline=tuple(),  # populated by upstream code; empty here
            authority_snapshot=AuthoritySnapshot.default(),
            output_loss_cumulative_gdp=0.0,  # populated downstream
            provenance=tuple(provenance_parts),
            data_quality=dq,
            notes=label.notes,
        )

    def assemble_many(
        self,
        labels: Iterable[CrisisLabel],
        series_loader: "MacroSeriesLoader",
    ) -> tuple[list[CrisisCase], list[tuple[CrisisLabel, str]]]:
        """Assemble many cases, returning (successes, failures)."""
        ok: list[CrisisCase] = []
        bad: list[tuple[CrisisLabel, str]] = []
        # Cache series by country to avoid reloading
        cache: dict[str, MacroSeries] = {}
        for label in labels:
            try:
                if label.country_iso3 not in cache:
                    cache[label.country_iso3] = series_loader.load_country(
                        label.country_iso3,
                    )
                series = cache[label.country_iso3]
                case = self.assemble_one(label, series)
                if case.data_quality < self.min_data_quality:
                    bad.append((
                        label,
                        f"data_quality {case.data_quality:.3f} < "
                        f"min {self.min_data_quality}",
                    ))
                    continue
                ok.append(case)
            except (DataPipelineError, ValueError) as exc:
                bad.append((label, f"{type(exc).__name__}: {exc}"))
                continue
        return ok, bad


# ---------------------------------------------------------------------------
# DataPipeline orchestrator
# ---------------------------------------------------------------------------


class DataPipeline:
    """End-to-end orchestrator: from raw CSVs to a populated ``CaseLibrary``.

    Usage
    -----
    Two-phase workflow:

    1. **Fit phase** (run once on training data)::

        pipeline = DataPipeline(...)
        library = pipeline.fit_build(
            crisis_csv="data/crises.csv",
            macro_dir="data/macro",
        )
        pipeline.save("artefacts/pipeline.json")
        library.save("artefacts/library")

    2. **Apply phase** (run on new/test data with the same statistics)::

        pipeline = DataPipeline.load("artefacts/pipeline.json")
        library = pipeline.apply_build(
            crisis_csv="data/test_crises.csv",
            macro_dir="data/macro",
        )

    The normaliser is fit once in phase 1 and frozen thereafter.
    """

    def __init__(
        self, *,
        feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
        pre_quarters: int = DEFAULT_PRE_QUARTERS,
        post_quarters: int = DEFAULT_POST_QUARTERS,
        max_gap_quarters: int = DEFAULT_MAX_GAP_QUARTERS,
        winsorize_pctiles: Optional[tuple[float, float]] = DEFAULT_WINSORIZE_PCTILES,
        min_data_quality: float = DEFAULT_MIN_DATA_QUALITY,
        derive: bool = True,
        normalizer: Optional[MacroFeatureNormalizer] = None,
    ) -> None:
        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self.pre_quarters: int = int(pre_quarters)
        self.post_quarters: int = int(post_quarters)
        self.max_gap_quarters: int = int(max_gap_quarters)
        self.winsorize_pctiles: Optional[tuple[float, float]] = (
            tuple(winsorize_pctiles) if winsorize_pctiles else None
        )
        self.min_data_quality: float = float(min_data_quality)
        self.derive: bool = bool(derive)
        if normalizer is None:
            self.normalizer = MacroFeatureNormalizer(
                feature_names=self.feature_names,
            )
        else:
            if normalizer.feature_names != self.feature_names:
                raise DataValidationError(
                    "DataPipeline: provided normalizer.feature_names "
                    f"{normalizer.feature_names} != "
                    f"feature_names {self.feature_names}"
                )
            self.normalizer = normalizer

    def fit_build(
        self,
        *,
        crisis_csv: Path,
        macro_dir: Path,
    ) -> CaseLibrary:
        """Fit normaliser on training data, then build the case library.

        Side-effect: ``self.normalizer`` becomes fitted.
        """
        crisis_csv, macro_dir = Path(crisis_csv), Path(macro_dir)
        labels_loader = CrisisLabelLoader(crisis_csv)
        labels = labels_loader.load()
        if not labels:
            raise DataPipelineError(
                f"DataPipeline.fit_build: no labels loaded from {crisis_csv}"
            )
        series_loader = MacroSeriesLoader(
            macro_dir, feature_names=self.feature_names,
            max_gap_quarters=self.max_gap_quarters,
            winsorize_pctiles=self.winsorize_pctiles, derive=self.derive,
        )

        # Phase 1: fit normaliser from all pre-onset windows of all labels
        pre_windows: list[np.ndarray] = []
        loaded_series: dict[str, MacroSeries] = {}
        for label in labels:
            if label.country_iso3 not in loaded_series:
                try:
                    loaded_series[label.country_iso3] = \
                        series_loader.load_country(label.country_iso3)
                except DataPipelineError as exc:
                    logger.warning(
                        "DataPipeline.fit_build: skipping %s: %s",
                        label.country_iso3, exc,
                    )
                    continue
            series = loaded_series[label.country_iso3]
            onset_idx = series.find_quarter_index(
                label.onset_year, label.onset_quarter,
            )
            if onset_idx is None:
                continue
            pre_start = onset_idx - self.pre_quarters
            if pre_start < 0:
                continue
            pre = series.data[pre_start:onset_idx]
            if np.all(np.isfinite(pre)):
                pre_windows.append(pre)

        if not pre_windows:
            raise DataPipelineError(
                "DataPipeline.fit_build: no valid pre-onset windows to fit "
                "the normaliser; check max_gap_quarters and pre_quarters"
            )
        X_fit = np.stack(pre_windows, axis=0)  # (N, T_pre, F)
        self.normalizer.fit(X_fit)
        logger.info(
            "DataPipeline.fit_build: fit normalizer on %d windows "
            "(%d quarters, %d features); fingerprint=%s",
            X_fit.shape[0], X_fit.shape[1], X_fit.shape[2],
            self.normalizer.fingerprint(),
        )

        # Phase 2: assemble
        assembler = CaseAssembler(
            pre_quarters=self.pre_quarters,
            post_quarters=self.post_quarters,
            normalizer=self.normalizer,
            feature_names=self.feature_names,
            min_data_quality=self.min_data_quality,
            labels_loader_path=crisis_csv,
            series_loader_dir=macro_dir,
        )
        ok, bad = assembler.assemble_many(labels, series_loader)
        for label, reason in bad:
            logger.warning(
                "DataPipeline.fit_build: skipped %s (%s): %s",
                label.case_id(), label.crisis_type, reason,
            )
        if not ok:
            raise DataPipelineError(
                "DataPipeline.fit_build: no cases survived assembly"
            )
        logger.info(
            "DataPipeline.fit_build: assembled %d cases; %d skipped",
            len(ok), len(bad),
        )
        return CaseLibrary(ok)

    def apply_build(
        self,
        *,
        crisis_csv: Path,
        macro_dir: Path,
    ) -> CaseLibrary:
        """Build a library using a previously-fit normaliser.

        Does not modify ``self.normalizer``.
        """
        if not self.normalizer.is_fitted:
            raise NormalizerNotFittedError(
                "DataPipeline.apply_build: normaliser is not fitted yet. "
                "Use fit_build() first, or set self.normalizer from a "
                "pre-fitted one."
            )
        crisis_csv, macro_dir = Path(crisis_csv), Path(macro_dir)
        labels_loader = CrisisLabelLoader(crisis_csv)
        labels = labels_loader.load()
        series_loader = MacroSeriesLoader(
            macro_dir, feature_names=self.feature_names,
            max_gap_quarters=self.max_gap_quarters,
            winsorize_pctiles=self.winsorize_pctiles, derive=self.derive,
        )
        assembler = CaseAssembler(
            pre_quarters=self.pre_quarters,
            post_quarters=self.post_quarters,
            normalizer=self.normalizer,
            feature_names=self.feature_names,
            min_data_quality=self.min_data_quality,
            labels_loader_path=crisis_csv,
            series_loader_dir=macro_dir,
        )
        ok, bad = assembler.assemble_many(labels, series_loader)
        for label, reason in bad:
            logger.warning(
                "DataPipeline.apply_build: skipped %s: %s",
                label.case_id(), reason,
            )
        if not ok:
            raise DataPipelineError(
                "DataPipeline.apply_build: no cases survived assembly"
            )
        return CaseLibrary(ok)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "feature_names": list(self.feature_names),
            "pre_quarters": int(self.pre_quarters),
            "post_quarters": int(self.post_quarters),
            "max_gap_quarters": int(self.max_gap_quarters),
            "winsorize_pctiles": (
                list(self.winsorize_pctiles)
                if self.winsorize_pctiles else None
            ),
            "min_data_quality": float(self.min_data_quality),
            "derive": bool(self.derive),
            "normalizer_is_fitted": self.normalizer.is_fitted,
            "normalizer_mean": (
                self.normalizer.mean.tolist()
                if self.normalizer.is_fitted else None
            ),
            "normalizer_std": (
                self.normalizer.std.tolist()
                if self.normalizer.is_fitted else None
            ),
            "normalizer_fingerprint": self.normalizer.fingerprint(),
            "saved_at": _utc_now_iso(),
        }
        _atomic_write_text(
            path,
            json.dumps(manifest, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "DataPipeline":
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _check_compatible_version(
            manifest.get("schema_version", SCHEMA_VERSION),
            context=f"DataPipeline@{path}",
        )
        feature_names = tuple(manifest["feature_names"])
        normalizer = MacroFeatureNormalizer(feature_names=feature_names)
        if manifest.get("normalizer_is_fitted") and \
                manifest.get("normalizer_mean") is not None:
            normalizer._mean = np.asarray(
                manifest["normalizer_mean"], dtype=np.float64,
            )
            normalizer._std = np.asarray(
                manifest["normalizer_std"], dtype=np.float64,
            )
            normalizer._fit_at = "<loaded from pipeline manifest>"
            # Fingerprint sanity
            expected = manifest.get("normalizer_fingerprint")
            if expected and normalizer.fingerprint() != expected:
                raise DataPipelineError(
                    f"DataPipeline.load@{path}: normaliser fingerprint "
                    f"mismatch (expected {expected}, got "
                    f"{normalizer.fingerprint()})"
                )
        return cls(
            feature_names=feature_names,
            pre_quarters=int(manifest["pre_quarters"]),
            post_quarters=int(manifest["post_quarters"]),
            max_gap_quarters=int(manifest["max_gap_quarters"]),
            winsorize_pctiles=(
                tuple(manifest["winsorize_pctiles"])
                if manifest.get("winsorize_pctiles") else None
            ),
            min_data_quality=float(manifest["min_data_quality"]),
            derive=bool(manifest["derive"]),
            normalizer=normalizer,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "feature_names": list(self.feature_names),
            "pre_quarters": int(self.pre_quarters),
            "post_quarters": int(self.post_quarters),
            "max_gap_quarters": int(self.max_gap_quarters),
            "winsorize_pctiles": (
                list(self.winsorize_pctiles)
                if self.winsorize_pctiles else None
            ),
            "min_data_quality": float(self.min_data_quality),
            "derive": bool(self.derive),
            "normalizer_is_fitted": self.normalizer.is_fitted,
            "normalizer_fingerprint": self.normalizer.fingerprint(),
        }

    def __repr__(self) -> str:
        return (
            f"DataPipeline(F={len(self.feature_names)}, "
            f"pre={self.pre_quarters}, post={self.post_quarters}, "
            f"normalizer_fitted={self.normalizer.is_fitted})"
        )


# ---------------------------------------------------------------------------
# Synthetic library generator (used by all upstream test suites)
# ---------------------------------------------------------------------------


#: Canonical type-center matrix used by ``build_synthetic_library``.
#: Each row is a (F=12)-dim mean vector for one of the five crisis types.
#: Coordinates are loosely calibrated to evoke the qualitative
#: signature of each crisis type without claiming statistical fidelity:
#: e.g. currency crises have a strong negative shift in ``reer_log_dev``
#: and a positive shift in ``sovereign_spread_bp``; sovereign crises
#: have an even stronger spread shift; triple crises combine all three.
_SYNTHETIC_TYPE_CENTERS: Final[dict[str, np.ndarray]] = {
    "banking":   np.array([0.0, 0.0, 0.0, 3.0, -2.0, 0.0, -1.0, 0.0,
                           -1.0, -1.0, 0.5, 0.0]),
    "currency":  np.array([0.0, 1.0, -2.0, 0.0, 0.0, 2.0, -3.0, 1.0,
                           0.0, 0.0, 0.0, 0.5]),
    "sovereign": np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 4.0, -1.0, 0.0,
                           -2.0, 0.0, 0.0, 0.5]),
    "twin":      np.array([-1.0, 1.0, -2.0, 1.0, -1.0, 3.0, -2.0, 1.0,
                           -1.0, 0.0, 0.0, 0.5]),
    "triple":    np.array([-2.0, 2.0, -3.0, 2.0, -3.0, 5.0, -3.0, 2.0,
                           -2.0, -1.0, 0.5, 1.0]),
}

#: Canonical countries cycled by ``build_synthetic_library``.
_SYNTHETIC_COUNTRIES: Final[tuple[str, ...]] = (
    "ARG", "BRA", "CHL", "COL", "EGY", "GHA", "HUN", "IDN",
    "IND", "KOR", "LBN", "LKA", "MEX", "PER", "PHL", "POL",
    "SAU", "THA", "TUR", "ZAF", "CHN", "MYS", "TWN", "GRC", "MAR",
)


def build_synthetic_library(
    *,
    n_per_type: int = 5,
    noise_std: float = 0.3,
    crisis_types: Sequence[str] = CRISIS_TYPES,
    seed: int = 2024,
    pre_quarters: int = DEFAULT_PRE_QUARTERS,
    post_quarters: int = DEFAULT_POST_QUARTERS,
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
    include_policy_timeline: bool = True,
) -> CaseLibrary:
    """Build a deterministic synthetic ``CaseLibrary`` for testing.

    Each crisis type contributes ``n_per_type`` cases, with pre-onset
    trajectories that linearly ramp from zero to the type's center
    vector, plus i.i.d. Gaussian noise. Post-onset trajectories
    fluctuate around half the center vector. The result is a clean,
    well-separated five-class problem that all upstream modules
    expect.

    This replaces the boilerplate that previously appeared in every
    test file. The synthetic centers, country list, and feature names
    are module-level constants for reproducibility across modules.

    Parameters
    ----------
    n_per_type : int
        Number of cases per crisis type.
    noise_std : float
        Gaussian noise std added to each quarterly observation.
    crisis_types : Sequence[str]
        Subset of ``case_memory.CRISIS_TYPES`` to include.
    seed : int
    pre_quarters, post_quarters : int
    feature_names : tuple[str, ...]
    include_policy_timeline : bool
        If True, attach a 1-entry policy timeline to each case.

    Returns
    -------
    library : CaseLibrary
    """
    if n_per_type < 1:
        raise DataValidationError(
            f"build_synthetic_library: n_per_type must be >= 1; "
            f"got {n_per_type}"
        )
    if noise_std < 0:
        raise DataValidationError(
            f"build_synthetic_library: noise_std must be >= 0; "
            f"got {noise_std}"
        )
    for ct in crisis_types:
        if ct not in _SYNTHETIC_TYPE_CENTERS:
            raise DataValidationError(
                f"build_synthetic_library: no synthetic center for "
                f"crisis_type {ct!r}; available: "
                f"{list(_SYNTHETIC_TYPE_CENTERS)}"
            )
    if len(feature_names) != N_MACRO_FEATURES:
        raise DataValidationError(
            f"build_synthetic_library: feature_names has length "
            f"{len(feature_names)} != N_MACRO_FEATURES "
            f"({N_MACRO_FEATURES})"
        )

    rng = np.random.default_rng(int(seed))
    cases: list[CrisisCase] = []
    i = 0
    for ct in crisis_types:
        center = _SYNTHETIC_TYPE_CENTERS[ct]
        for k in range(n_per_type):
            ctry = _SYNTHETIC_COUNTRIES[i % len(_SYNTHETIC_COUNTRIES)]
            yr = 1995 + (i * 3) % 28
            q = (i % 4) + 1
            pre = np.zeros((pre_quarters, N_MACRO_FEATURES), dtype=np.float64)
            for t in range(pre_quarters):
                pre[t] = (
                    (t / max(1, pre_quarters - 1)) * center
                    + rng.standard_normal(N_MACRO_FEATURES) * noise_std
                )
            post = np.zeros(
                (post_quarters, N_MACRO_FEATURES), dtype=np.float64,
            )
            for t in range(post_quarters):
                post[t] = (
                    center * 0.5
                    + rng.standard_normal(N_MACRO_FEATURES) * noise_std
                )
            if include_policy_timeline:
                policy_timeline: tuple[PolicyAction, ...] = (
                    PolicyAction(
                        date=f"{yr}-{q*3:02d}-15",
                        institution=INSTITUTIONS[0],
                        lever="policy_rate",
                        value=200.0 + 50.0 * i,
                        units="bp_change",
                        note="synthetic",
                    ),
                )
            else:
                policy_timeline = tuple()
            cases.append(CrisisCase(
                case_id=f"{ctry}_{yr}_Q{q}",
                country_iso3=ctry,
                onset_year=yr, onset_quarter=q,
                crisis_type=ct,
                pre_onset_trajectory=pre,
                post_onset_trajectory=post,
                feature_names=tuple(feature_names),
                policy_timeline=policy_timeline,
                authority_snapshot=AuthoritySnapshot.default(),
                # Cycle output_loss in [5, 150] to stay safely under
                # case_memory's plausibility cap (|loss| <= 200) at any
                # n_per_type. The 5-step cycle gives 30 distinct values
                # before wrapping, plenty for synthetic variation.
                output_loss_cumulative_gdp=float(5.0 + (i % 30) * 5.0),
                provenance=("build_synthetic_library",
                            f"seed={seed}",
                            f"noise_std={noise_std}"),
                data_quality=0.9,
                notes=f"synthetic case for type={ct}",
                # Fixed created_at so two calls with the same seed
                # produce bit-identical libraries (otherwise
                # content_checksum drifts across processes, breaking
                # the analogy engine's strict library-checksum check
                # at load time).
                created_at="2024-01-01T00:00:00+00:00",
            ))
            i += 1
    return CaseLibrary(cases)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> int:
    """CLI: ``python data_pipeline.py <command> [args]``.

    Commands
    --------
    summary <pipeline_json>
        Load and print a summary of a saved DataPipeline manifest.
    synthetic <output_dir> [--n-per-type N] [--seed S]
        Build a synthetic case library and save it to ``output_dir``.
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="python data_pipeline.py",
        description="Inspect and operate the data pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summary", help="Print pipeline summary.")
    p_sum.add_argument("pipeline_json", type=Path)

    p_syn = sub.add_parser(
        "synthetic", help="Build a synthetic library and save it.",
    )
    p_syn.add_argument("output_dir", type=Path)
    p_syn.add_argument("--n-per-type", type=int, default=5)
    p_syn.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _configure_logging(args.log_level)

    try:
        if args.command == "summary":
            pipeline = DataPipeline.load(args.pipeline_json)
            print(json.dumps(pipeline.summary(), indent=2, sort_keys=True))
            return 0
        elif args.command == "synthetic":
            lib = build_synthetic_library(
                n_per_type=args.n_per_type, seed=args.seed,
            )
            lib.save(args.output_dir)
            print(
                f"Built synthetic library with {len(lib)} cases at "
                f"{args.output_dir}"
            )
            return 0
        else:  # pragma: no cover
            parser.print_help()
            return 2
    except CaseMemoryError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
