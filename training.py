"""
training.py
===========

Five-phase training orchestrator.

This module ties together the five trainable sub-systems of the
retrieval-augmented multi-agent crisis-mitigation framework into a
single reproducible training run that produces a complete set of
artefacts consumable by ``evaluation.py``, ``experiments.py``, and
``mitigation.MultiAgentMitigationPolicy.get_action``.

Phase graph
-----------
::

    Phase A: DetectorEnsemble.fit(library)
                        │
                        ▼   detector probabilities on training cases
    Phase B1: CoordinatorRouter.fit(detector_probs, macro_states, labels)
                        │
                        │   (Phase B1 and B2 can run in parallel; their
                        │    outputs are independent until Phase C consumes
                        │    the signature encoder.)
                        ▼
    Phase B2: CaseSignatureEncoder.fit(library)
                        │
                        ▼   trained signature encoder
    Phase C: AnalogyEngine.fit() with
            (signature_encoder, ConditionalRetriever, CaseContextEncoder)
                        │
                        ▼   trained analogy engine
    Phase D: MitigationTrainer rollout + TD3 updates with
            (MultiAgentMitigationPolicy, LinearCrisisDynamics,
            AnalogyEngine, ReplayBuffer)
                        │
                        ▼
    [Phase E (optional): joint fine-tuning of all modules — stub for
     extension; not exercised in the camera-ready run.]

Each phase is independently restartable. If a phase's artefact
directory already exists with a valid manifest, ``run_all_phases``
loads it and skips training; otherwise the phase trains from scratch
and persists. This lets the user re-run Phase D with a different
``case_coherence_weight`` (for ablation) without retraining Phases A
through C.

Reproducibility commitments
---------------------------
* Each phase receives a deterministically-derived seed
  (``config.seed + phase_offset[phase]``) that is recorded in the
  phase manifest.
* All RNG state restoration is via save-and-restore of the global
  torch/numpy generators around module construction.
* Phase artefacts are bit-identical save/load (verified end-to-end
  by the upstream modules' own test suites).
* Disk writes are atomic.
* The top-level training manifest records the library checksum, so
  rebuilding the library with new data forces a re-train of all
  phases.

Hardware
--------
The orchestrator is device-aware. Default device selection:

  * If ``config.device`` is explicitly set, honour it.
  * Else: ``"cuda"`` if available, otherwise ``"cpu"``.

The framework's model sizes (the largest is the mitigation policy at
~250k parameters) fit comfortably on a single GPU. Multi-GPU is not
required and not used. A 4× NVIDIA RTX 3090 setup will use device 0
unless the user specifies ``"cuda:1"``, ``"cuda:2"``, or ``"cuda:3"``.

References (APA-7)
------------------
Fujimoto, S., Hoof, H., & Meger, D. (2018). Addressing function
    approximation error in actor-critic methods. In Proceedings of
    the 35th International Conference on Machine Learning
    (Vol. 80, pp. 1587-1596).

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
    calibration of modern neural networks. In Proceedings of the
    34th International Conference on Machine Learning (Vol. 70,
    pp. 1321-1330).

Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P.,
    Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised
    contrastive learning. In Advances in Neural Information
    Processing Systems (Vol. 33, pp. 18661-18673).

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T.,
    Tassa, Y., Silver, D., & Wierstra, D. (2016). Continuous control
    with deep reinforcement learning. In International Conference on
    Learning Representations.

Oord, A. van den, Li, Y., & Vinyals, O. (2018). Representation
    learning with contrastive predictive coding. arXiv preprint
    arXiv:1807.03748.

Version
-------
1.0.0  Camera-ready KBS submission.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional, Union

import numpy as np
import torch

from case_memory import (
    CRISIS_TYPES,
    CaseLibrary,
    CaseMemoryError,
    CaseSignatureEncoder,
    DEFAULT_SIGNATURE_DIM,
    N_MACRO_FEATURES,
)
from analogy_engine import (
    AnalogyEngine,
    CaseContextEncoder,
    ConditionalRetriever,
    DEFAULT_CONTEXT_DIM,
    DEFAULT_K_RETRIEVAL,
    DEFAULT_METRIC_RANK,
    DEFAULT_RETRIEVAL_TEMPERATURE,
    compute_outcome_fingerprint,
    compute_policy_fingerprint,
)
from coordinator import (
    COORDINATOR_TYPES,
    CoordinatorRouter,
)
from detection import DetectorEnsemble
from mitigation import (
    AuthorityGraph,
    ControlBarrierFunction,
    JOINT_ACTION_DIM,
    LinearCrisisDynamics,
    MitigationTransition,
    MitigationTrainer,
    MultiAgentMitigationPolicy,
    ReplayBuffer,
    SafetyBounds,
    collapse_24dim_fp_to_per_lever,
    compute_reward,
)
from case_memory import AuthoritySnapshot


__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "PHASE_NAMES",
    "PHASE_SEED_OFFSETS",
    # Exceptions
    "TrainingError",
    "PhaseError",
    "ArtifactNotFoundError",
    # Dataclasses
    "TrainingConfig",
    "PhaseResult",
    "TrainingArtifacts",
    # Orchestrator
    "TrainingOrchestrator",
    # Convenience
    "resolve_device",
    "compute_library_checksum",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Semantic version of the on-disk training artefacts.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: Phase identifiers used as subdirectory names under ``output_dir``.
PHASE_NAMES: Final[tuple[str, ...]] = (
    "phase_a_detection",
    "phase_b1_coordinator",
    "phase_b2_signature",
    "phase_c_analogy",
    "phase_d_mitigation",
)

#: Deterministic per-phase seed offsets so that re-running one phase
#: with a different seed doesn't disturb other phases' randomness.
PHASE_SEED_OFFSETS: Final[dict[str, int]] = {
    "phase_a_detection":     1_000,
    "phase_b1_coordinator":  2_000,
    "phase_b2_signature":    3_000,
    "phase_c_analogy":       4_000,
    "phase_d_mitigation":    5_000,
}


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class TrainingError(CaseMemoryError):
    """Base class for training-orchestrator exceptions."""


class PhaseError(TrainingError):
    """A specific phase failed during training."""


class ArtifactNotFoundError(TrainingError):
    """A required artefact (e.g., from a prior phase) was missing
    on disk and could not be auto-built."""


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


def _check_compatible_version(
    found_version: str,
    expected_version: str = SCHEMA_VERSION,
    context: str = "object",
) -> None:
    try:
        found = tuple(int(p) for p in found_version.split("."))
        expected = tuple(int(p) for p in expected_version.split("."))
    except (ValueError, AttributeError) as exc:
        raise TrainingError(
            f"{context}: malformed schema_version {found_version!r}; "
            f"expected MAJOR.MINOR.PATCH"
        ) from exc
    if len(found) != 3 or len(expected) != 3:
        raise TrainingError(
            f"{context}: schema_version must be MAJOR.MINOR.PATCH; "
            f"got {found_version!r}"
        )
    if found[0] != expected[0]:
        raise TrainingError(
            f"{context}: schema major-version mismatch (found "
            f"{found_version!r}, code supports {expected_version!r})"
        )
    if found[1] != expected[1]:
        logger.warning(
            "%s: schema minor-version mismatch (found %s, code supports %s)",
            context, found_version, expected_version,
        )


def _seed_everything(seed: int) -> None:
    """Set seeds for python's random, numpy, and torch."""
    import random
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**31 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def resolve_device(device_hint: Optional[str] = None) -> torch.device:
    """Resolve a device hint to a torch.device.

    Resolution order:
      * If ``device_hint`` is provided and CUDA is requested but
        unavailable, fall back to CPU with a warning.
      * If ``device_hint`` is ``None`` or "auto", use CUDA if available,
        otherwise CPU.
      * Otherwise, honour the explicit string.
    """
    if device_hint in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_hint.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "resolve_device: CUDA requested (%r) but not available; "
            "falling back to CPU.", device_hint,
        )
        return torch.device("cpu")
    return torch.device(device_hint)


def compute_library_checksum(library: CaseLibrary) -> str:
    """A short, deterministic checksum identifying a library.

    Used by the training manifest so that re-running training with a
    different library invalidates prior phases automatically (the user
    must explicitly clear the output directory to retrain).
    """
    case_ids = sorted(library.case_ids())
    parts = []
    for cid in case_ids:
        case = library[cid]
        parts.append(
            f"{cid}|{case.crisis_type}|{case.onset_year}Q{case.onset_quarter}"
        )
    digest = _sha256_hex("\n".join(parts))[:16]
    return f"{len(case_ids)}cases:{digest}"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Full training configuration.

    Every hyperparameter for every phase lives here so a single
    serialised config fully determines a training run (modulo
    hardware nondeterminism in CUDA kernels, which is mitigated by
    deterministic seeding but cannot be eliminated entirely).

    Defaults are tuned for the canonical 25-50 case library and
    target the "Setup A" reported in the manuscript. Override
    individual fields for ablation runs.

    Common
    ------
    seed : int
        Global seed; per-phase seeds are derived deterministically.
    device : str
        Torch device; ``"auto"`` picks CUDA if available else CPU.
    output_dir : Path
        Root directory for artefact persistence.
    verbose : bool
        If True, propagate verbose flags to phase trainers.

    Phase A (Detection)
    -------------------
    detection_n_epochs : int
    detection_lr : float
    detection_batch_size : int
    detection_val_fraction : float
    detection_n_negatives_per_detector : Optional[int]
        If None, defaults to len(library).
    detection_weight_decay : float
    detection_calibrate : bool

    Phase B1 (Coordinator)
    ----------------------
    coordinator_n_epochs : int
    coordinator_lr : float
    coordinator_batch_size : int
    coordinator_val_fraction : float
    coordinator_label_smoothing : float
    coordinator_weight_decay : float

    Phase B2 (Signature Encoder)
    ----------------------------
    signature_n_epochs : int
    signature_lr : float
    signature_batch_size : Optional[int]
        None means use full-batch training (typical for small libraries).
    signature_temperature : float
        SupCon temperature.
    signature_weight_decay : float

    Phase C (Analogy Engine)
    ------------------------
    retriever_rank : int
        Per-type metric rank.
    context_dim : int
        Context-vector dimensionality consumed by the policy.
    retriever_n_epochs : int
    retriever_lr : float
    retriever_temperature : float
    retriever_label_smoothing : float
    retriever_weight_decay : float

    Phase D (Mitigation)
    --------------------
    mitigation_n_episodes : int
        Number of episodes to roll out.
    mitigation_episode_len : int
        Steps per episode.
    mitigation_batch_size : int
    mitigation_buffer_capacity : int
    mitigation_warmup_steps : int
        Steps with random actions before TD3 updates start.
    mitigation_update_freq : int
        How often (in env steps) to call trainer.step.
    mitigation_actor_lr : float
    mitigation_critic_lr : float
    mitigation_gamma : float
    mitigation_polyak : float
    mitigation_case_coherence_weight : float
    mitigation_safety_penalty_weight : float
    mitigation_exploration_noise : float
    mitigation_k_retrieval : int
    mitigation_use_ground_truth_type : bool
        If True, condition retrieval on the case's ground-truth
        crisis type (training stability). If False, use a uniform
        prior (more realistic). Set True for camera-ready.

    Phase E (Joint fine-tuning) — stub for extension
    ------------------------------------------------
    joint_enabled : bool
        Default False; set True only if the user has an explicit
        fine-tuning recipe to apply (camera-ready does not).
    """
    # Common
    seed: int = 42
    device: str = "auto"
    output_dir: Path = field(default_factory=lambda: Path("./artifacts"))
    verbose: bool = False

    # Phase A
    detection_n_epochs: int = 200
    detection_lr: float = 1e-3
    detection_batch_size: int = 32
    detection_val_fraction: float = 0.15
    detection_n_negatives_per_detector: Optional[int] = None
    detection_weight_decay: float = 1e-4
    detection_calibrate: bool = True

    # Phase B1
    coordinator_n_epochs: int = 200
    coordinator_lr: float = 1e-3
    coordinator_batch_size: int = 32
    coordinator_val_fraction: float = 0.15
    coordinator_label_smoothing: float = 0.05
    coordinator_weight_decay: float = 1e-4

    # Phase B2
    signature_n_epochs: int = 200
    signature_lr: float = 1e-3
    signature_batch_size: Optional[int] = None
    signature_temperature: float = 0.1
    signature_weight_decay: float = 1e-4

    # Phase C
    signature_dim: int = DEFAULT_SIGNATURE_DIM
    context_dim: int = DEFAULT_CONTEXT_DIM
    retriever_rank: int = DEFAULT_METRIC_RANK
    retriever_n_epochs: int = 200
    retriever_lr: float = 1e-3
    retriever_temperature: float = DEFAULT_RETRIEVAL_TEMPERATURE
    retriever_label_smoothing: float = 0.1
    retriever_weight_decay: float = 1e-4

    # Phase D
    mitigation_n_episodes: int = 200
    mitigation_episode_len: int = 12
    mitigation_batch_size: int = 64
    mitigation_buffer_capacity: int = 10_000
    mitigation_warmup_steps: int = 500
    mitigation_update_freq: int = 1
    mitigation_actor_lr: float = 3e-4
    mitigation_critic_lr: float = 3e-4
    mitigation_gamma: float = 0.95
    mitigation_polyak: float = 0.995
    mitigation_case_coherence_weight: float = 0.1
    mitigation_safety_penalty_weight: float = 1.0
    mitigation_exploration_noise: float = 0.1
    mitigation_k_retrieval: int = DEFAULT_K_RETRIEVAL
    mitigation_use_ground_truth_type: bool = True

    # Phase E (stub)
    joint_enabled: bool = False

    def __post_init__(self) -> None:
        # Coerce output_dir to Path
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        # Validate ranges
        if self.seed < 0:
            raise TrainingError(f"seed must be >= 0; got {self.seed}")
        if not 0.0 <= self.detection_val_fraction < 1.0:
            raise TrainingError(
                f"detection_val_fraction must be in [0, 1); "
                f"got {self.detection_val_fraction}"
            )
        if not 0.0 <= self.coordinator_val_fraction < 1.0:
            raise TrainingError(
                f"coordinator_val_fraction must be in [0, 1); "
                f"got {self.coordinator_val_fraction}"
            )
        if self.signature_dim <= 0:
            raise TrainingError(
                f"signature_dim must be > 0; got {self.signature_dim}"
            )
        if self.context_dim <= 0:
            raise TrainingError(
                f"context_dim must be > 0; got {self.context_dim}"
            )
        if self.retriever_rank <= 0:
            raise TrainingError(
                f"retriever_rank must be > 0; got {self.retriever_rank}"
            )
        if self.mitigation_n_episodes < 0:
            raise TrainingError(
                f"mitigation_n_episodes must be >= 0; "
                f"got {self.mitigation_n_episodes}"
            )
        if self.mitigation_episode_len <= 0:
            raise TrainingError(
                f"mitigation_episode_len must be > 0; "
                f"got {self.mitigation_episode_len}"
            )
        if self.mitigation_warmup_steps < 0:
            raise TrainingError(
                f"mitigation_warmup_steps must be >= 0; "
                f"got {self.mitigation_warmup_steps}"
            )
        if self.mitigation_buffer_capacity <= 0:
            raise TrainingError(
                f"mitigation_buffer_capacity must be > 0; "
                f"got {self.mitigation_buffer_capacity}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        d = dataclasses.asdict(self)
        # Convert Path to str
        d["output_dir"] = str(self.output_dir)
        return d

    def fingerprint(self) -> str:
        """SHA-256 short digest. Two configs with identical fields
        produce the same fingerprint."""
        payload = self.to_dict()
        # Exclude output_dir (path doesn't affect training outcome)
        payload.pop("output_dir", None)
        return _sha256_hex(_canonical_json(payload))[:16]


@dataclass(frozen=True)
class PhaseResult:
    """Per-phase outcome record stored in the training manifest."""
    phase_name: str
    status: str  # "complete" | "loaded" | "skipped"
    started_at: str
    finished_at: str
    duration_seconds: float
    seed: int
    artefact_path: str
    metrics: dict[str, Any]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": float(self.duration_seconds),
            "seed": int(self.seed),
            "artefact_path": str(self.artefact_path),
            "metrics": dict(self.metrics),
            "notes": str(self.notes),
        }


@dataclass
class TrainingArtifacts:
    """Container for all trained sub-modules.

    Populated incrementally as phases complete. Phases that have not
    yet run have ``None`` entries.
    """
    detection_ensemble: Optional[DetectorEnsemble] = None
    coordinator: Optional[CoordinatorRouter] = None
    signature_encoder: Optional[CaseSignatureEncoder] = None
    analogy_engine: Optional[AnalogyEngine] = None
    mitigation_policy: Optional[MultiAgentMitigationPolicy] = None
    dynamics: Optional[LinearCrisisDynamics] = None
    phase_results: list[PhaseResult] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return all(x is not None for x in (
            self.detection_ensemble, self.coordinator,
            self.signature_encoder, self.analogy_engine,
            self.mitigation_policy, self.dynamics,
        ))


# ---------------------------------------------------------------------------
# TrainingOrchestrator
# ---------------------------------------------------------------------------


class TrainingOrchestrator:
    """End-to-end five-phase training driver.

    Usage
    -----
    ::

        from training import TrainingConfig, TrainingOrchestrator
        from data_pipeline import build_synthetic_library

        library = build_synthetic_library(n_per_type=5)
        config = TrainingConfig(
            output_dir=Path("./artifacts/run_001"),
            seed=42, verbose=True,
        )
        orchestrator = TrainingOrchestrator(config, library)
        artefacts = orchestrator.run_all_phases()

    Each phase artefact is persisted to a subdirectory of
    ``config.output_dir``. The top-level manifest at
    ``output_dir/training_manifest.json`` records what completed
    and with what metrics. Re-running ``run_all_phases`` on the same
    output directory loads existing phase artefacts and skips
    re-training; clear the directory (or delete individual phase
    sub-directories) to force retraining.
    """

    def __init__(
        self,
        config: TrainingConfig,
        library: CaseLibrary,
    ) -> None:
        if not isinstance(config, TrainingConfig):
            raise TrainingError(
                f"config must be a TrainingConfig; got {type(config)}"
            )
        if not isinstance(library, CaseLibrary):
            raise TrainingError(
                f"library must be a CaseLibrary; got {type(library)}"
            )
        if len(library) == 0:
            raise TrainingError(
                "TrainingOrchestrator: library is empty"
            )
        self.config: TrainingConfig = config
        self.library: CaseLibrary = library
        self.library_checksum: str = compute_library_checksum(library)
        self.device: torch.device = resolve_device(config.device)
        self.artefacts: TrainingArtifacts = TrainingArtifacts()
        self._output_dir: Path = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _phase_seed(self, phase_name: str) -> int:
        """Deterministic per-phase seed."""
        if phase_name not in PHASE_SEED_OFFSETS:
            raise TrainingError(
                f"_phase_seed: unknown phase {phase_name!r}"
            )
        return int(self.config.seed) + PHASE_SEED_OFFSETS[phase_name]

    def _phase_dir(self, phase_name: str) -> Path:
        if phase_name not in PHASE_NAMES:
            raise TrainingError(
                f"_phase_dir: unknown phase {phase_name!r}"
            )
        return self._output_dir / phase_name

    def _record_phase(self, result: PhaseResult) -> None:
        """Append a PhaseResult to artefacts and write the manifest."""
        self.artefacts.phase_results.append(result)
        self._save_manifest()

    @staticmethod
    def _sanitise_for_json(obj: Any) -> Any:
        """Recursively replace non-finite floats with None so the manifest
        can be serialised with ``allow_nan=False``.

        We use ``allow_nan=False`` because non-finite values in a
        reproducibility-critical artefact are almost always a bug, and
        silent NaN-propagation in downstream evaluation would be worse
        than the explicit null. The replacement here is a defensive
        last resort; the orchestrator strives to never produce NaN
        in the first place.
        """
        import math
        if isinstance(obj, float):
            return None if not math.isfinite(obj) else obj
        if isinstance(obj, dict):
            return {k: TrainingOrchestrator._sanitise_for_json(v)
                    for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [TrainingOrchestrator._sanitise_for_json(v) for v in obj]
        return obj

    def _save_manifest(self) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "library_checksum": self.library_checksum,
            "library_size": int(len(self.library)),
            "config": self.config.to_dict(),
            "config_fingerprint": self.config.fingerprint(),
            "device": str(self.device),
            "phase_results": [r.to_dict() for r in self.artefacts.phase_results],
            "is_complete": bool(self.artefacts.is_complete),
            "saved_at": _utc_now_iso(),
        }
        manifest = self._sanitise_for_json(manifest)
        _atomic_write_text(
            self._output_dir / "training_manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )

    # ------------------------------------------------------------------ #
    # Phase A: Detection
    # ------------------------------------------------------------------ #

    def train_phase_a_detection(
        self, *, force_retrain: bool = False,
    ) -> DetectorEnsemble:
        """Train the four binary crisis detectors.

        Side-effect: populates ``self.artefacts.detection_ensemble``.
        If a saved ensemble exists at ``self._phase_dir("phase_a_detection")``
        and ``force_retrain`` is False, loads from disk and returns.
        """
        phase_name = "phase_a_detection"
        phase_dir = self._phase_dir(phase_name)
        started_at = _utc_now_iso()
        t0 = time.time()
        seed = self._phase_seed(phase_name)

        # Try to load
        if not force_retrain and (phase_dir / "ensemble.pt").is_file():
            logger.info(
                "Phase A: loading existing ensemble from %s", phase_dir,
            )
            ensemble = DetectorEnsemble.load(
                phase_dir, map_location=self.device,
            )
            self.artefacts.detection_ensemble = ensemble
            self._record_phase(PhaseResult(
                phase_name=phase_name, status="loaded",
                started_at=started_at,
                finished_at=_utc_now_iso(),
                duration_seconds=time.time() - t0,
                seed=seed,
                artefact_path=str(phase_dir),
                metrics={"loaded_from_disk": True},
                notes="loaded from cached artefacts",
            ))
            return ensemble

        # Train from scratch
        logger.info("Phase A: training detection ensemble (seed=%d)", seed)
        ensemble = DetectorEnsemble(
            n_features=N_MACRO_FEATURES,
            seed=seed,
        )
        ensemble.to(self.device)
        history = ensemble.fit(
            self.library,
            n_negatives_per_detector=self.config.detection_n_negatives_per_detector,
            val_fraction=self.config.detection_val_fraction,
            n_epochs=self.config.detection_n_epochs,
            lr=self.config.detection_lr,
            weight_decay=self.config.detection_weight_decay,
            batch_size=self.config.detection_batch_size,
            calibrate=self.config.detection_calibrate,
            seed=seed,
            verbose=self.config.verbose,
        )
        ensemble.save(phase_dir)

        # Per-detector final metrics
        per_detector_metrics: dict[str, Any] = {}
        for name, hist in history.items():
            per_detector_metrics[name] = {
                "final_loss": (
                    float(hist["loss"][-1]) if hist.get("loss") else None
                ),
                "final_accuracy": (
                    float(hist["accuracy"][-1])
                    if hist.get("accuracy") else None
                ),
                "calibration_temperature": (
                    float(hist["temperature"][-1])
                    if hist.get("temperature") else None
                ),
                "val_pre_nll": (
                    float(hist["val_pre_nll"][-1])
                    if hist.get("val_pre_nll") else None
                ),
                "val_post_nll": (
                    float(hist["val_post_nll"][-1])
                    if hist.get("val_post_nll") else None
                ),
            }

        self.artefacts.detection_ensemble = ensemble
        self._record_phase(PhaseResult(
            phase_name=phase_name, status="complete",
            started_at=started_at,
            finished_at=_utc_now_iso(),
            duration_seconds=time.time() - t0,
            seed=seed,
            artefact_path=str(phase_dir),
            metrics={"per_detector": per_detector_metrics},
        ))
        return ensemble

    # ------------------------------------------------------------------ #
    # Phase B1: Coordinator
    # ------------------------------------------------------------------ #

    def train_phase_b1_coordinator(
        self,
        ensemble: Optional[DetectorEnsemble] = None,
        *,
        force_retrain: bool = False,
    ) -> CoordinatorRouter:
        """Train the coordinator on top of the detection ensemble.

        Constructs per-case detector probabilities from the ensemble's
        forward pass, builds macro-state and label arrays, then fits
        the coordinator with stratified train/val split for calibration.
        """
        phase_name = "phase_b1_coordinator"
        phase_dir = self._phase_dir(phase_name)
        started_at = _utc_now_iso()
        t0 = time.time()
        seed = self._phase_seed(phase_name)

        if ensemble is None:
            ensemble = self.artefacts.detection_ensemble
        if ensemble is None:
            raise PhaseError(
                "Phase B1: detection ensemble not available; run Phase A "
                "first or pass `ensemble` explicitly."
            )

        # Try to load
        if not force_retrain and (phase_dir / "router.pt").is_file():
            logger.info(
                "Phase B1: loading existing coordinator from %s", phase_dir,
            )
            router = CoordinatorRouter.load(
                phase_dir, map_location=self.device,
            )
            self.artefacts.coordinator = router
            self._record_phase(PhaseResult(
                phase_name=phase_name, status="loaded",
                started_at=started_at,
                finished_at=_utc_now_iso(),
                duration_seconds=time.time() - t0,
                seed=seed,
                artefact_path=str(phase_dir),
                metrics={"loaded_from_disk": True},
                notes="loaded from cached artefacts",
            ))
            return router

        # Build training arrays from the ensemble's forward pass
        logger.info("Phase B1: building coordinator training arrays")
        case_ids = self.library.case_ids()
        n = len(case_ids)
        detector_probs = np.zeros((n, 4), dtype=np.float64)
        macro_states = np.zeros((n, N_MACRO_FEATURES), dtype=np.float64)
        labels = np.zeros(n, dtype=np.int64)
        for i, cid in enumerate(case_ids):
            case = self.library[cid]
            # Get detector probabilities
            ens_out = ensemble.detect(case.pre_onset_trajectory)
            detector_probs[i] = ens_out.probabilities
            # Last quarter of pre-onset trajectory = current macro state
            macro_states[i] = case.pre_onset_trajectory[-1]
            # Crisis type index in COORDINATOR_TYPES
            if case.crisis_type not in COORDINATOR_TYPES:
                raise PhaseError(
                    f"Phase B1: case {cid} has crisis_type "
                    f"{case.crisis_type!r} not in COORDINATOR_TYPES "
                    f"{list(COORDINATOR_TYPES)}"
                )
            labels[i] = COORDINATOR_TYPES.index(case.crisis_type)

        # Stratified train/val split
        rng = np.random.default_rng(seed)
        all_idx = np.arange(n)
        if self.config.coordinator_val_fraction > 0:
            # Stratified by label
            train_idx: list[int] = []
            val_idx: list[int] = []
            unique_labels = np.unique(labels)
            for lab in unique_labels:
                lab_idx = all_idx[labels == lab]
                rng.shuffle(lab_idx)
                n_val = max(
                    1,
                    int(round(
                        self.config.coordinator_val_fraction * len(lab_idx),
                    )),
                ) if len(lab_idx) > 1 else 0
                val_idx.extend(lab_idx[:n_val].tolist())
                train_idx.extend(lab_idx[n_val:].tolist())
            train_idx_arr = np.asarray(train_idx, dtype=np.int64)
            val_idx_arr = np.asarray(val_idx, dtype=np.int64)
            X_train_p = detector_probs[train_idx_arr]
            X_train_s = macro_states[train_idx_arr]
            y_train = labels[train_idx_arr]
            X_val_p = detector_probs[val_idx_arr] if len(val_idx_arr) > 0 else None
            X_val_s = macro_states[val_idx_arr] if len(val_idx_arr) > 0 else None
            y_val = labels[val_idx_arr] if len(val_idx_arr) > 0 else None
        else:
            X_train_p, X_train_s, y_train = detector_probs, macro_states, labels
            X_val_p = X_val_s = y_val = None

        logger.info(
            "Phase B1: training coordinator on %d train + %d val (seed=%d)",
            X_train_p.shape[0], (X_val_p.shape[0] if X_val_p is not None else 0),
            seed,
        )
        router = CoordinatorRouter(
            macro_state_dim=N_MACRO_FEATURES, seed=seed,
        )
        router.to(self.device)
        hist = router.fit(
            X_train_p, X_train_s, y_train,
            val_detector_probs=X_val_p,
            val_macro_states=X_val_s,
            val_labels=y_val,
            n_epochs=self.config.coordinator_n_epochs,
            lr=self.config.coordinator_lr,
            weight_decay=self.config.coordinator_weight_decay,
            batch_size=self.config.coordinator_batch_size,
            label_smoothing=self.config.coordinator_label_smoothing,
            seed=seed,
            verbose=self.config.verbose,
        )
        router.save(phase_dir)

        metrics = {
            "n_train": int(X_train_p.shape[0]),
            "n_val": int(X_val_p.shape[0]) if X_val_p is not None else 0,
            "final_loss": (
                float(hist["loss"][-1])
                if hist.get("loss") else None
            ),
            "final_accuracy": (
                float(hist["accuracy"][-1])
                if hist.get("accuracy") else None
            ),
            # Coordinator val metrics: pre-calibration NLL and
            # post-calibration NLL (computed once after LBFGS
            # temperature optimisation; see Guo et al. 2017).
            "val_pre_nll": (
                float(hist["val_pre_nll"][-1])
                if hist.get("val_pre_nll") else None
            ),
            "val_post_nll": (
                float(hist["val_post_nll"][-1])
                if hist.get("val_post_nll") else None
            ),
            "calibration_temperature": (
                float(hist["temperature"][-1])
                if hist.get("temperature") else None
            ),
        }
        self.artefacts.coordinator = router
        self._record_phase(PhaseResult(
            phase_name=phase_name, status="complete",
            started_at=started_at,
            finished_at=_utc_now_iso(),
            duration_seconds=time.time() - t0,
            seed=seed,
            artefact_path=str(phase_dir),
            metrics=metrics,
        ))
        return router

    # ------------------------------------------------------------------ #
    # Phase B2: Signature Encoder
    # ------------------------------------------------------------------ #

    def train_phase_b2_signature(
        self, *, force_retrain: bool = False,
    ) -> CaseSignatureEncoder:
        """Train the signature encoder via supervised contrastive loss.

        Phase B2 is independent of Phase B1 (the signature encoder and
        coordinator consume different inputs and produce different
        outputs). They can be trained in any order after Phase A.
        """
        phase_name = "phase_b2_signature"
        phase_dir = self._phase_dir(phase_name)
        started_at = _utc_now_iso()
        t0 = time.time()
        seed = self._phase_seed(phase_name)

        if not force_retrain and (phase_dir / "encoder.pt").is_file():
            logger.info(
                "Phase B2: loading existing signature encoder from %s",
                phase_dir,
            )
            encoder = CaseSignatureEncoder.load(
                phase_dir, map_location=self.device,
            )
            self.artefacts.signature_encoder = encoder
            self._record_phase(PhaseResult(
                phase_name=phase_name, status="loaded",
                started_at=started_at,
                finished_at=_utc_now_iso(),
                duration_seconds=time.time() - t0,
                seed=seed,
                artefact_path=str(phase_dir),
                metrics={"loaded_from_disk": True},
                notes="loaded from cached artefacts",
            ))
            return encoder

        logger.info(
            "Phase B2: training signature encoder (seed=%d, dim=%d)",
            seed, self.config.signature_dim,
        )
        encoder = CaseSignatureEncoder(
            n_features=N_MACRO_FEATURES,
            signature_dim=self.config.signature_dim,
            seed=seed,
        )
        encoder.to(self.device)
        hist = encoder.fit(
            self.library,
            n_epochs=self.config.signature_n_epochs,
            lr=self.config.signature_lr,
            weight_decay=self.config.signature_weight_decay,
            batch_size=self.config.signature_batch_size,
            temperature=self.config.signature_temperature,
            device=self.device,
            verbose=self.config.verbose,
        )
        encoder.save(phase_dir)

        metrics = {
            "n_epochs_actual": len(hist.get("loss", [])),
            "final_loss": (
                float(hist["loss"][-1]) if hist.get("loss") else None
            ),
            "min_loss": (
                float(min(hist["loss"])) if hist.get("loss") else None
            ),
        }
        self.artefacts.signature_encoder = encoder
        self._record_phase(PhaseResult(
            phase_name=phase_name, status="complete",
            started_at=started_at,
            finished_at=_utc_now_iso(),
            duration_seconds=time.time() - t0,
            seed=seed,
            artefact_path=str(phase_dir),
            metrics=metrics,
        ))
        return encoder

    # ------------------------------------------------------------------ #
    # Phase C: Analogy Engine
    # ------------------------------------------------------------------ #

    def train_phase_c_analogy(
        self,
        signature_encoder: Optional[CaseSignatureEncoder] = None,
        *,
        force_retrain: bool = False,
    ) -> AnalogyEngine:
        """Train the conditional retriever and case context encoder.

        Constructs the analogy engine with the trained signature
        encoder, then calls its joint fit method. The context encoder's
        fit (separate from the analogy engine's fit) computes the
        per-feature standardisers for outcome and policy fingerprints —
        this is a closed-form statistics fit, not gradient training.
        """
        phase_name = "phase_c_analogy"
        phase_dir = self._phase_dir(phase_name)
        started_at = _utc_now_iso()
        t0 = time.time()
        seed = self._phase_seed(phase_name)

        if signature_encoder is None:
            signature_encoder = self.artefacts.signature_encoder
        if signature_encoder is None:
            raise PhaseError(
                "Phase C: signature encoder not available; run Phase B2 "
                "first or pass `signature_encoder` explicitly."
            )

        if not force_retrain and (phase_dir / "engine_manifest.json").is_file():
            logger.info(
                "Phase C: loading existing analogy engine from %s",
                phase_dir,
            )
            engine = AnalogyEngine.load(
                phase_dir, library=self.library,
                signature_encoder=signature_encoder,
                device=self.device,
            )
            self.artefacts.analogy_engine = engine
            self._record_phase(PhaseResult(
                phase_name=phase_name, status="loaded",
                started_at=started_at,
                finished_at=_utc_now_iso(),
                duration_seconds=time.time() - t0,
                seed=seed,
                artefact_path=str(phase_dir),
                metrics={"loaded_from_disk": True},
                notes="loaded from cached artefacts",
            ))
            return engine

        logger.info(
            "Phase C: training analogy engine (seed=%d, rank=%d, "
            "context_dim=%d)",
            seed, self.config.retriever_rank, self.config.context_dim,
        )
        retriever = ConditionalRetriever(
            signature_dim=self.config.signature_dim,
            n_types=len(CRISIS_TYPES),
            rank=self.config.retriever_rank,
            seed=seed,
        )
        context_encoder = CaseContextEncoder(
            signature_dim=self.config.signature_dim,
            context_dim=self.config.context_dim,
            seed=seed + 1,
        )
        # Fit context encoder (statistics, not gradient training)
        ce_stats = context_encoder.fit(self.library)
        logger.info(
            "Phase C: context encoder fit complete: %s",
            {k: (v if not isinstance(v, np.ndarray) else f"<array shape={v.shape}>")
             for k, v in ce_stats.items()},
        )

        # Build the analogy engine
        engine = AnalogyEngine(
            self.library, signature_encoder, retriever, context_encoder,
            device=self.device,
        )
        # Train the retriever's W matrices
        hist = engine.fit(
            n_epochs=self.config.retriever_n_epochs,
            lr=self.config.retriever_lr,
            weight_decay=self.config.retriever_weight_decay,
            temperature=self.config.retriever_temperature,
            label_smoothing=self.config.retriever_label_smoothing,
            verbose=self.config.verbose,
        )
        engine.save(phase_dir)

        # AnalogyEngine.fit returns only ['epoch', 'loss']. Retrieval
        # recall metrics are computed by evaluation.py against held-out
        # cases, not during training. This module's responsibility ends
        # at fit-loss convergence.
        metrics = {
            "n_epochs_actual": len(hist.get("loss", [])),
            "final_loss": (
                float(hist["loss"][-1]) if hist.get("loss") else None
            ),
            "min_loss": (
                float(min(hist["loss"])) if hist.get("loss") else None
            ),
        }
        self.artefacts.analogy_engine = engine
        self._record_phase(PhaseResult(
            phase_name=phase_name, status="complete",
            started_at=started_at,
            finished_at=_utc_now_iso(),
            duration_seconds=time.time() - t0,
            seed=seed,
            artefact_path=str(phase_dir),
            metrics=metrics,
        ))
        return engine

    # ------------------------------------------------------------------ #
    # Phase D: Mitigation
    # ------------------------------------------------------------------ #

    def train_phase_d_mitigation(
        self,
        analogy_engine: Optional[AnalogyEngine] = None,
        *,
        force_retrain: bool = False,
    ) -> tuple[MultiAgentMitigationPolicy, LinearCrisisDynamics]:
        """Train the case-augmented multi-agent mitigation policy via TD3.

        The training loop:
          1. Fit a ``LinearCrisisDynamics`` model on the library
             (closed-form ridge regression).
          2. Build the ``MultiAgentMitigationPolicy`` with default
             ``AuthorityGraph`` and ``ControlBarrierFunction``.
          3. Roll out ``mitigation_n_episodes`` episodes of length
             ``mitigation_episode_len``. Each step: encode the current
             trajectory window via the signature encoder, retrieve
             k cases, sample an action, step dynamics, append a
             ``MitigationTransition`` to the replay buffer.
          4. After warmup, take a TD3 step at every ``update_freq``
             environment steps.

        Returns
        -------
        policy : MultiAgentMitigationPolicy
        dynamics : LinearCrisisDynamics
        """
        phase_name = "phase_d_mitigation"
        phase_dir = self._phase_dir(phase_name)
        started_at = _utc_now_iso()
        t0 = time.time()
        seed = self._phase_seed(phase_name)

        if analogy_engine is None:
            analogy_engine = self.artefacts.analogy_engine
        if analogy_engine is None:
            raise PhaseError(
                "Phase D: analogy engine not available; run Phase C "
                "first or pass `analogy_engine` explicitly."
            )

        if not force_retrain and (phase_dir / "policy_manifest.json").is_file():
            logger.info(
                "Phase D: loading existing mitigation policy from %s",
                phase_dir,
            )
            policy = MultiAgentMitigationPolicy.load(
                phase_dir, map_location=self.device,
            )
            # Reload dynamics. LinearCrisisDynamics.save() creates two
            # files inside the target directory: dynamics.pt (state
            # dict) and dynamics_config.json (architecture config).
            dyn_path = phase_dir / "dynamics"
            if dyn_path.is_dir() and (dyn_path / "dynamics_config.json").is_file():
                dynamics = LinearCrisisDynamics.load(dyn_path)
            else:
                # Re-fit on the library as a fallback. This branch
                # should not be reached if the phase completed
                # successfully; warn if hit.
                logger.warning(
                    "Phase D: dynamics not found at %s; refitting", dyn_path,
                )
                dynamics = LinearCrisisDynamics(seed=seed)
                dynamics.fit(self.library)
            self.artefacts.mitigation_policy = policy
            self.artefacts.dynamics = dynamics
            self._record_phase(PhaseResult(
                phase_name=phase_name, status="loaded",
                started_at=started_at,
                finished_at=_utc_now_iso(),
                duration_seconds=time.time() - t0,
                seed=seed,
                artefact_path=str(phase_dir),
                metrics={"loaded_from_disk": True},
                notes="loaded from cached artefacts",
            ))
            return policy, dynamics

        logger.info(
            "Phase D: training mitigation policy "
            "(seed=%d, n_episodes=%d, episode_len=%d)",
            seed, self.config.mitigation_n_episodes,
            self.config.mitigation_episode_len,
        )

        # 1. Fit dynamics
        dynamics = LinearCrisisDynamics(seed=seed)
        dyn_stats = dynamics.fit(self.library, verbose=self.config.verbose)
        logger.info(
            "Phase D: dynamics fit: spectral_radius=%.4f, "
            "n_pairs=%d, residual_norm=%.4f",
            dyn_stats.get("spectral_radius_A", float("nan")),
            int(dyn_stats.get("n_pairs", 0)),
            dyn_stats.get("residual_norm", float("nan")),
        )

        # 2. Build policy
        authority = AuthorityGraph(AuthoritySnapshot.default())
        safety_bounds = SafetyBounds()
        cbf = ControlBarrierFunction(safety_bounds, authority)
        policy = MultiAgentMitigationPolicy(
            authority, cbf=cbf,
            context_dim=self.config.context_dim,
            n_crisis_types=len(CRISIS_TYPES),
            seed=seed,
        )
        policy.to(self.device)

        # 3. Build replay buffer and trainer
        buffer = ReplayBuffer(
            capacity=self.config.mitigation_buffer_capacity,
            context_dim=self.config.context_dim,
            n_crisis_types=len(CRISIS_TYPES),
            k_retrieval=self.config.mitigation_k_retrieval,
            seed=seed + 1,
        )
        trainer = MitigationTrainer(
            policy,
            actor_lr=self.config.mitigation_actor_lr,
            critic_lr=self.config.mitigation_critic_lr,
            gamma=self.config.mitigation_gamma,
            polyak=self.config.mitigation_polyak,
            case_coherence_weight=self.config.mitigation_case_coherence_weight,
            safety_penalty_weight=self.config.mitigation_safety_penalty_weight,
        )

        # 4. Rollout loop
        env_rng = np.random.default_rng(seed + 2)
        update_step = 0
        n_total_env_steps = (
            self.config.mitigation_n_episodes
            * self.config.mitigation_episode_len
        )
        train_metrics: dict[str, list[float]] = {
            "critic_loss": [],
            "actor_loss": [],
            "case_coherence_loss": [],
            "safety_penalty": [],
            "mean_episode_reward": [],
        }
        case_ids = self.library.case_ids()
        logger.info(
            "Phase D: starting rollout (%d total env steps, warmup=%d, "
            "device=%s)",
            n_total_env_steps, self.config.mitigation_warmup_steps,
            self.device,
        )
        global_env_step = 0
        for episode in range(self.config.mitigation_n_episodes):
            # Sample a starting case
            cid = case_ids[env_rng.integers(0, len(case_ids))]
            case = self.library[cid]
            # State = last quarter of pre-onset trajectory
            state = case.pre_onset_trajectory[-1].astype(np.float64)
            # Use the full pre-onset trajectory as the retrieval query
            query_traj = case.pre_onset_trajectory.copy()
            # Type posterior: ground truth if requested, else uniform
            if self.config.mitigation_use_ground_truth_type:
                tp = np.zeros(len(CRISIS_TYPES))
                tp[CRISIS_TYPES.index(case.crisis_type)] = 1.0
            else:
                tp = np.ones(len(CRISIS_TYPES)) / len(CRISIS_TYPES)
            episode_rewards: list[float] = []

            for step in range(self.config.mitigation_episode_len):
                # Retrieve
                result = analogy_engine.retrieve(
                    query_traj, tp,
                    k=self.config.mitigation_k_retrieval,
                    temperature=self.config.retriever_temperature,
                )
                # Build per-lever fingerprints and outcomes
                fps = np.stack([
                    collapse_24dim_fp_to_per_lever(
                        compute_policy_fingerprint(self.library[rcid])
                    )
                    for rcid in result.case_ids
                ], axis=0).astype(np.float64)  # (K, 8)
                outcomes = np.asarray([
                    compute_outcome_fingerprint(self.library[rcid])[0]
                    for rcid in result.case_ids
                ], dtype=np.float64)  # (K,)

                # Sample action
                if global_env_step < self.config.mitigation_warmup_steps:
                    # Random action during warmup
                    action_unit = env_rng.uniform(
                        -1.0, 1.0, size=(JOINT_ACTION_DIM,),
                    )
                    action_info: dict[str, Any] = {"warmup": True}
                else:
                    action_unit, action_info = policy.get_action(
                        state, result,
                        exploration_noise=self.config.mitigation_exploration_noise,
                        apply_cbf=True,
                        generator=env_rng,
                    )

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

                # Reward and done
                reward = compute_reward(state, action_unit)
                done = (step == self.config.mitigation_episode_len - 1)
                episode_rewards.append(reward)

                # Build transition
                transition = MitigationTransition(
                    state=state,
                    joint_action_unit=action_unit,
                    reward=float(reward),
                    next_state=next_state,
                    done=bool(done),
                    context_vector=result.context_vector,
                    type_posterior=result.type_posterior,
                    retrieved_fp_per_lever=fps,
                    retrieved_outcomes=outcomes,
                    retrieval_weights=result.weights,
                )
                buffer.append(transition)
                global_env_step += 1

                # TD3 update
                if (
                    global_env_step >= self.config.mitigation_warmup_steps
                    and global_env_step % self.config.mitigation_update_freq == 0
                    and len(buffer) >= self.config.mitigation_batch_size
                ):
                    batch = buffer.sample(
                        self.config.mitigation_batch_size,
                        device=self.device,
                    )
                    step_metrics = trainer.step(batch)
                    # critic_loss is reported every step
                    cl = step_metrics.get("critic_loss")
                    if cl is not None:
                        train_metrics["critic_loss"].append(float(cl))
                    # actor_loss is None on skipped steps (TD3 delayed
                    # actor update at frequency `actor_update_freq`),
                    # so guard with `is not None` rather than `in`.
                    al = step_metrics.get("actor_loss")
                    if al is not None:
                        train_metrics["actor_loss"].append(float(al))
                    # MitigationTrainer.step reports the case-coherence
                    # term under the key "case_coherence" (not
                    # "case_coherence_loss"); same None-skip semantics
                    # as actor_loss above.
                    cc = step_metrics.get("case_coherence")
                    if cc is not None:
                        train_metrics["case_coherence_loss"].append(float(cc))
                    # safety_penalty: also None on skip
                    sp = step_metrics.get("safety_penalty")
                    if sp is not None:
                        train_metrics["safety_penalty"].append(float(sp))
                    update_step += 1

                # Update query trajectory rolling window (shift left by 1)
                query_traj = np.concatenate(
                    [query_traj[1:], next_state[np.newaxis, :]], axis=0,
                )

                # Advance state
                state = next_state

            train_metrics["mean_episode_reward"].append(
                float(np.mean(episode_rewards))
            )
            if self.config.verbose and (episode + 1) % 20 == 0:
                recent_critic = (
                    np.mean(train_metrics["critic_loss"][-20:])
                    if train_metrics["critic_loss"] else float("nan")
                )
                logger.info(
                    "Phase D: episode %d/%d  mean_R=%.3f  recent_critic_loss=%.4f",
                    episode + 1, self.config.mitigation_n_episodes,
                    train_metrics["mean_episode_reward"][-1],
                    recent_critic,
                )

        # 5. Persist
        policy.save(phase_dir)
        dynamics.save(phase_dir / "dynamics")

        metrics = {
            "n_episodes": self.config.mitigation_n_episodes,
            "n_env_steps": global_env_step,
            "n_update_steps": update_step,
            "final_mean_episode_reward": (
                float(np.mean(train_metrics["mean_episode_reward"][-20:]))
                if train_metrics["mean_episode_reward"] else None
            ),
            "final_critic_loss": (
                float(np.mean(train_metrics["critic_loss"][-50:]))
                if train_metrics["critic_loss"] else None
            ),
            "final_actor_loss": (
                float(np.mean(train_metrics["actor_loss"][-50:]))
                if train_metrics["actor_loss"] else None
            ),
            "final_case_coherence_loss": (
                float(np.mean(train_metrics["case_coherence_loss"][-50:]))
                if train_metrics["case_coherence_loss"] else None
            ),
            "final_safety_penalty": (
                float(np.mean(train_metrics["safety_penalty"][-50:]))
                if train_metrics["safety_penalty"] else None
            ),
            "dynamics_spectral_radius": float(
                dyn_stats.get("spectral_radius_A", float("nan"))
            ),
            "dynamics_residual_norm": float(
                dyn_stats.get("residual_norm", float("nan"))
            ),
            "dynamics_n_pairs": int(
                dyn_stats.get("n_pairs", 0)
            ),
        }
        self.artefacts.mitigation_policy = policy
        self.artefacts.dynamics = dynamics
        self._record_phase(PhaseResult(
            phase_name=phase_name, status="complete",
            started_at=started_at,
            finished_at=_utc_now_iso(),
            duration_seconds=time.time() - t0,
            seed=seed,
            artefact_path=str(phase_dir),
            metrics=metrics,
        ))
        return policy, dynamics

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def run_all_phases(
        self, *, force_retrain: bool = False,
    ) -> TrainingArtifacts:
        """Run all five phases in order.

        If a phase's artefacts exist on disk and ``force_retrain`` is
        False, load instead of train. Returns the populated artefacts
        object.
        """
        logger.info(
            "TrainingOrchestrator: starting full run; output_dir=%s, "
            "device=%s, library_checksum=%s",
            self._output_dir, self.device, self.library_checksum,
        )
        # Persist a top-level manifest before training so users can
        # see config even on early failure.
        self._save_manifest()

        ensemble = self.train_phase_a_detection(force_retrain=force_retrain)
        _ = self.train_phase_b1_coordinator(
            ensemble, force_retrain=force_retrain,
        )
        signature_encoder = self.train_phase_b2_signature(
            force_retrain=force_retrain,
        )
        analogy_engine = self.train_phase_c_analogy(
            signature_encoder, force_retrain=force_retrain,
        )
        _ = self.train_phase_d_mitigation(
            analogy_engine, force_retrain=force_retrain,
        )

        if self.config.joint_enabled:
            logger.info(
                "Phase E (joint fine-tuning) is requested but the "
                "camera-ready recipe does not exercise it; skipping."
            )

        # Final manifest write
        self._save_manifest()
        logger.info(
            "TrainingOrchestrator: full run complete; artefacts at %s",
            self._output_dir,
        )
        return self.artefacts

    def summary(self) -> dict[str, Any]:
        """Lightweight summary suitable for logging or display."""
        return {
            "schema_version": SCHEMA_VERSION,
            "library_checksum": self.library_checksum,
            "library_size": int(len(self.library)),
            "device": str(self.device),
            "config_fingerprint": self.config.fingerprint(),
            "output_dir": str(self._output_dir),
            "phase_status": {
                r.phase_name: r.status
                for r in self.artefacts.phase_results
            },
            "is_complete": bool(self.artefacts.is_complete),
        }

    def __repr__(self) -> str:
        n_phases = len(self.artefacts.phase_results)
        return (
            f"TrainingOrchestrator(library={self.library_checksum}, "
            f"device={self.device}, phases_done={n_phases}/5, "
            f"complete={self.artefacts.is_complete})"
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
    """CLI: ``python training.py <command> [args]``.

    Commands
    --------
    synthetic <output_dir> [--n-per-type N] [--seed S]
        Run a complete training session on a freshly-built synthetic
        library. Smoke-test entry point.
    summary <manifest_path>
        Print the training manifest at the given path.
    """
    import argparse
    from data_pipeline import build_synthetic_library

    parser = argparse.ArgumentParser(
        prog="python training.py",
        description="Train all five sub-systems of the framework.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_syn = sub.add_parser(
        "synthetic", help="Train on a freshly-built synthetic library.",
    )
    p_syn.add_argument("output_dir", type=Path)
    p_syn.add_argument("--n-per-type", type=int, default=5)
    p_syn.add_argument("--seed", type=int, default=42)
    p_syn.add_argument("--device", default="auto")
    p_syn.add_argument(
        "--quick", action="store_true",
        help="Use small epoch counts and episode counts for a smoke test.",
    )

    p_sum = sub.add_parser("summary", help="Print a training manifest.")
    p_sum.add_argument("manifest_path", type=Path)

    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    _configure_logging(args.log_level)

    try:
        if args.command == "synthetic":
            library = build_synthetic_library(
                n_per_type=args.n_per_type, seed=args.seed,
            )
            cfg_kwargs: dict[str, Any] = dict(
                output_dir=args.output_dir,
                seed=args.seed,
                device=args.device,
                verbose=False,
            )
            if args.quick:
                cfg_kwargs.update(dict(
                    detection_n_epochs=20,
                    coordinator_n_epochs=20,
                    signature_n_epochs=20,
                    retriever_n_epochs=20,
                    mitigation_n_episodes=10,
                    mitigation_episode_len=4,
                    mitigation_warmup_steps=10,
                ))
            config = TrainingConfig(**cfg_kwargs)
            orchestrator = TrainingOrchestrator(config, library)
            orchestrator.run_all_phases()
            print(json.dumps(orchestrator.summary(), indent=2, sort_keys=True))
            return 0
        elif args.command == "summary":
            manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        else:  # pragma: no cover
            parser.print_help()
            return 2
    except (CaseMemoryError, TrainingError) as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
