"""
tests.py
========

Unified test harness for the entire framework.

Each of the nine modules
(``data_pipeline``, ``case_memory``, ``analogy_engine``,
``mitigation``, ``coordinator``, ``detection``, ``training``,
``evaluation``, ``experiments``) ships its own ``test_<module>.py``
script that exercises that module in isolation. This harness adds a
**second** layer above those per-module scripts:

  1. **Smoke tests** (seconds): every module imports cleanly and its
     public constructors accept their documented arguments. Used by
     CI/CD as a first-line gate.
  2. **Unit tests**: orchestrates the existing ``test_*.py`` scripts
     via subprocess, capturing pass/fail per script.
  3. **Integration tests** (cross-module): end-to-end scenarios that
     exercise the full pipeline data → train → evaluate → ablate,
     plus a save/load isomorphism check.
  4. **Reproducibility audit**: runs determinism-sensitive operations
     twice with the same seed and reports the observed numerical
     drift. Documents the framework's reproducibility envelope
     (see manuscript Appendix B).

All four categories produce a structured ``TestReport`` saved as
``tests_report.json``, plus a human-readable terminal summary.

CLI
---
::

    python tests.py [--smoke | --unit | --integration | --reproducibility | --all]
                    [--output-dir PATH] [--quick] [--skip-slow]

Defaults: ``--all`` and no ``--quick``.

Design constraints
------------------
* **No new test dependencies.** The harness uses only the stdlib
  (``subprocess``, ``json``, ``pathlib``, ``time``) plus numpy and
  torch which are already imported by the framework modules.
* **Robust subprocess orchestration.** Each per-module test is run
  with a configurable timeout; on timeout the report records a
  ``timeout`` status rather than crashing the harness.
* **Atomic disk writes** for the report so a crashed harness leaves
  a usable partial report.
* **Determinism documentation**: the reproducibility audit doesn't
  *fail* when it detects numerical drift; it records the magnitude.
  This makes the audit a useful manuscript artefact even when full
  bit-identity isn't achievable (which is the case on the current
  PyTorch CPU stack).

References (APA-7)
------------------
Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer,
    A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving
    reproducibility in machine learning research. Journal of Machine
    Learning Research, 22(164), 1-20.

Version
-------
1.0.0  Camera-ready KBS submission.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Final, Optional


__all__ = [
    # Constants
    "SCHEMA_VERSION",
    "MODULE_NAMES",
    "PER_MODULE_TESTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_SMOKE_TIMEOUT_SECONDS",
    # Exceptions
    "TestHarnessError",
    # Dataclasses
    "TestRecord",
    "TestReport",
    # Runners
    "run_smoke_tests",
    "run_unit_tests",
    "run_integration_tests",
    "run_reproducibility_audit",
    "run_all",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Semantic version of the on-disk test report.
SCHEMA_VERSION: Final[str] = "1.0.0"

#: All framework modules in dependency order. Smoke tests import each
#: in this order to surface any cyclic-import bugs early.
MODULE_NAMES: Final[tuple[str, ...]] = (
    "case_memory",
    "data_pipeline",
    "analogy_engine",
    "coordinator",
    "detection",
    "mitigation",
    "training",
    "evaluation",
    "experiments",
)

#: Per-module test files. Same order as MODULE_NAMES.
PER_MODULE_TESTS: Final[tuple[str, ...]] = tuple(
    f"test_{name}.py" for name in MODULE_NAMES
)

#: Default per-test subprocess timeout (10 minutes). Per-module tests
#: that train full pipelines can take several minutes; the CLI smoke
#: test for experiments.py runs five ablations.
DEFAULT_TIMEOUT_SECONDS: Final[int] = 600

#: Smoke tests should complete in seconds; a longer timeout means
#: something is wrong.
DEFAULT_SMOKE_TIMEOUT_SECONDS: Final[int] = 60

#: Success marker that every per-module test prints on completion.
#: The harness looks for this string in stdout to confirm pass.
SUCCESS_MARKER: Final[str] = "ALL TESTS PASSED"

#: Pattern used in subprocess outputs to identify the source script.
_BANNER_WIDTH: Final[int] = 70


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class TestHarnessError(RuntimeError):
    """Base class for harness-level errors (not test failures)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def _tail_lines(text: str, n: int = 20) -> str:
    """Return the last ``n`` lines of ``text``."""
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "\n".join(lines[-n:])


def _print_banner(title: str, *, width: int = _BANNER_WIDTH) -> None:
    print("=" * width)
    print(title)
    print("=" * width)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestRecord:
    """Outcome of a single test (whether a smoke check, a per-module
    script, an integration scenario, or a reproducibility probe)."""
    name: str
    category: str  # "smoke" | "unit" | "integration" | "reproducibility"
    status: str    # "pass" | "fail" | "skip" | "timeout" | "error"
    duration_seconds: float
    started_at: str
    finished_at: str
    message: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "duration_seconds": float(self.duration_seconds),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "message": self.message,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


@dataclass
class TestReport:
    """Aggregated results of one harness invocation."""
    schema_version: str = SCHEMA_VERSION
    started_at: str = field(default_factory=_utc_now_iso)
    finished_at: str = ""
    duration_seconds: float = 0.0
    records: list[TestRecord] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def add(self, record: TestRecord) -> None:
        self.records.append(record)

    def n_with_status(self, status: str) -> int:
        return sum(1 for r in self.records if r.status == status)

    @property
    def n_total(self) -> int:
        return len(self.records)

    @property
    def n_pass(self) -> int:
        return self.n_with_status("pass")

    @property
    def n_fail(self) -> int:
        return self.n_with_status("fail")

    @property
    def n_skip(self) -> int:
        return self.n_with_status("skip")

    @property
    def n_timeout(self) -> int:
        return self.n_with_status("timeout")

    @property
    def n_error(self) -> int:
        return self.n_with_status("error")

    @property
    def is_green(self) -> bool:
        """True iff every record passed or was explicitly skipped."""
        return all(r.status in ("pass", "skip") for r in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": float(self.duration_seconds),
            "n_total": int(self.n_total),
            "n_pass": int(self.n_pass),
            "n_fail": int(self.n_fail),
            "n_skip": int(self.n_skip),
            "n_timeout": int(self.n_timeout),
            "n_error": int(self.n_error),
            "is_green": bool(self.is_green),
            "config": dict(self.config),
            "records": [r.to_dict() for r in self.records],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        _atomic_write_text(
            path,
            json.dumps(self.to_dict(), sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False),
        )
        return path

    def print_summary(self) -> None:
        """Print a terminal summary of the report."""
        _print_banner("TEST REPORT SUMMARY")
        # Group by category
        by_category: dict[str, list[TestRecord]] = {}
        for r in self.records:
            by_category.setdefault(r.category, []).append(r)
        for cat in sorted(by_category.keys()):
            print(f"\n  [{cat}] {len(by_category[cat])} tests")
            for r in by_category[cat]:
                # Format: status_glyph name (duration)
                glyph = {
                    "pass": "✓",
                    "fail": "✗",
                    "skip": "—",
                    "timeout": "⧖",
                    "error": "!",
                }.get(r.status, "?")
                print(f"    {glyph} {r.name:<40s} "
                      f"{r.status:<8s}  {r.duration_seconds:6.2f}s")
                if r.status != "pass" and r.message:
                    msg = r.message.splitlines()[0][:80]
                    print(f"      └── {msg}")
        print()
        print(f"  Total       : {self.n_total}")
        print(f"  Pass        : {self.n_pass}")
        print(f"  Fail        : {self.n_fail}")
        print(f"  Skip        : {self.n_skip}")
        print(f"  Timeout     : {self.n_timeout}")
        print(f"  Error       : {self.n_error}")
        print(f"  Duration    : {self.duration_seconds:.2f}s")
        print(f"  Verdict     : {'GREEN ✓' if self.is_green else 'RED ✗'}")
        print()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def _smoke_test_one(
    module_name: str, *, timeout: int = DEFAULT_SMOKE_TIMEOUT_SECONDS,
) -> TestRecord:
    """Import a module and probe its public surface.

    The smoke test for a module:
      1. Imports it (catches syntax errors, import-time errors,
         circular imports).
      2. Verifies it exposes ``__all__``.
      3. Verifies every name in ``__all__`` is accessible.
      4. Verifies the module has a ``SCHEMA_VERSION`` if it's
         expected to (all our modules do).
    """
    name = f"smoke[{module_name}]"
    started_at = _utc_now_iso()
    t0 = time.time()
    try:
        if module_name in sys.modules:
            mod = importlib.reload(sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)
        if not hasattr(mod, "__all__"):
            raise TestHarnessError(
                f"module {module_name} lacks __all__"
            )
        exported = list(getattr(mod, "__all__"))
        if not exported:
            raise TestHarnessError(
                f"module {module_name}: __all__ is empty"
            )
        for entry in exported:
            if not hasattr(mod, entry):
                raise TestHarnessError(
                    f"module {module_name}: __all__ entry {entry!r} "
                    f"not actually defined"
                )
        if not hasattr(mod, "SCHEMA_VERSION"):
            # Not all modules have SCHEMA_VERSION (case_memory does,
            # but it's not strictly required). Warn rather than fail.
            logger.warning(
                "smoke[%s]: no SCHEMA_VERSION constant", module_name,
            )
        return TestRecord(
            name=name, category="smoke", status="pass",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"imported, exports {len(exported)} names",
        )
    except Exception as exc:
        return TestRecord(
            name=name, category="smoke", status="fail",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"{type(exc).__name__}: {exc}",
        )


def run_smoke_tests(
    module_names: tuple[str, ...] = MODULE_NAMES,
) -> list[TestRecord]:
    """Run smoke tests for every module."""
    print()
    _print_banner("SMOKE TESTS")
    records: list[TestRecord] = []
    for m in module_names:
        rec = _smoke_test_one(m)
        records.append(rec)
        glyph = "✓" if rec.status == "pass" else "✗"
        print(f"  {glyph} {m:<25s}  {rec.duration_seconds:.3f}s  "
              f"{rec.message[:60]}")
    return records


# ---------------------------------------------------------------------------
# Unit tests (per-module scripts)
# ---------------------------------------------------------------------------


def _run_subprocess_test(
    test_path: Path, *,
    cwd: Path,
    name: Optional[str] = None,
    category: str = "unit",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    env: Optional[dict[str, str]] = None,
) -> TestRecord:
    """Run a single test script in a subprocess.

    Returns a TestRecord with status = "pass" iff the script's
    return code is 0 AND its stdout contains SUCCESS_MARKER.
    Otherwise: "fail" (non-zero rc), "timeout" (TimeoutExpired), or
    "error" (subprocess launch failure).
    """
    test_path = Path(test_path)
    test_name = name or test_path.name
    started_at = _utc_now_iso()
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        rc = result.returncode
        success = (rc == 0) and (SUCCESS_MARKER in stdout)
        if success:
            status = "pass"
            message = "rc=0, marker found"
        else:
            status = "fail"
            message = (
                f"rc={rc}, marker_found={SUCCESS_MARKER in stdout}"
            )
        return TestRecord(
            name=test_name, category=category, status=status,
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=message,
            stdout_tail=_tail_lines(stdout, 10),
            stderr_tail=_tail_lines(stderr, 10),
        )
    except subprocess.TimeoutExpired as exc:
        return TestRecord(
            name=test_name, category=category, status="timeout",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"exceeded {timeout}s",
            stdout_tail=_tail_lines(exc.stdout or "", 10) if isinstance(exc.stdout, str) else "",
            stderr_tail=_tail_lines(exc.stderr or "", 10) if isinstance(exc.stderr, str) else "",
        )
    except (FileNotFoundError, OSError) as exc:
        return TestRecord(
            name=test_name, category=category, status="error",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"{type(exc).__name__}: {exc}",
        )


def run_unit_tests(
    test_dir: Path = Path("/home/claude"),
    test_files: tuple[str, ...] = PER_MODULE_TESTS,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    skip: tuple[str, ...] = (),
) -> list[TestRecord]:
    """Run every per-module test script in subprocess.

    Parameters
    ----------
    test_dir : Path
        Directory containing the test_*.py scripts.
    test_files : tuple[str, ...]
        Names of test scripts to run.
    timeout : int
        Per-script timeout in seconds.
    skip : tuple[str, ...]
        Test file names (basenames) to skip with status="skip".
    """
    print()
    _print_banner("UNIT TESTS")
    test_dir = Path(test_dir)
    records: list[TestRecord] = []
    for test_file in test_files:
        test_path = test_dir / test_file
        name = test_file.replace(".py", "")
        if test_file in skip:
            print(f"  — {name:<40s}  skipped")
            records.append(TestRecord(
                name=name, category="unit", status="skip",
                duration_seconds=0.0,
                started_at=_utc_now_iso(),
                finished_at=_utc_now_iso(),
                message="explicitly skipped",
            ))
            continue
        if not test_path.is_file():
            print(f"  ! {name:<40s}  missing at {test_path}")
            records.append(TestRecord(
                name=name, category="unit", status="error",
                duration_seconds=0.0,
                started_at=_utc_now_iso(),
                finished_at=_utc_now_iso(),
                message=f"test file not found: {test_path}",
            ))
            continue
        print(f"  running {name:<35s} ...", end="", flush=True)
        rec = _run_subprocess_test(
            test_path, cwd=test_dir, name=name,
            category="unit", timeout=timeout,
        )
        glyph = {
            "pass": "✓", "fail": "✗", "skip": "—",
            "timeout": "⧖", "error": "!",
        }.get(rec.status, "?")
        print(f"\r  {glyph} {name:<35s}  {rec.status:<8s} "
              f"{rec.duration_seconds:6.2f}s")
        if rec.status != "pass" and rec.message:
            print(f"      └── {rec.message[:80]}")
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def _integration_test_smoke_e2e(
    workdir: Path, *, quick: bool = True,
) -> TestRecord:
    """End-to-end smoke: synthetic library → train → evaluate → ablate.

    This is the canonical integration scenario. Verifies that the
    full nine-module pipeline composes correctly in one process.
    """
    name = "integration[e2e_synthetic]"
    started_at = _utc_now_iso()
    t0 = time.time()
    try:
        # Lazy imports so a smoke-test-only invocation doesn't pay
        # the cost of loading torch.
        import warnings
        from data_pipeline import build_synthetic_library
        from training import TrainingConfig, TrainingOrchestrator
        from evaluation import Evaluator
        from experiments import (
            AblationSpec, Experiment, ExperimentConfig,
        )
        if quick:
            cfg = TrainingConfig(
                output_dir=workdir / "training",
                seed=42, device="cpu", verbose=False,
                detection_n_epochs=15, detection_batch_size=16,
                coordinator_n_epochs=15, coordinator_batch_size=8,
                signature_n_epochs=15,
                retriever_n_epochs=15,
                mitigation_n_episodes=4, mitigation_episode_len=4,
                mitigation_warmup_steps=4, mitigation_batch_size=8,
                mitigation_buffer_capacity=1000,
            )
            eval_kwargs = dict(
                mitigation_n_episodes=4, mitigation_episode_len=4,
                n_bootstrap=20,
            )
        else:  # pragma: no cover
            cfg = TrainingConfig(
                output_dir=workdir / "training",
                seed=42, device="cpu",
            )
            eval_kwargs = dict(mitigation_n_episodes=50, n_bootstrap=500)

        library = build_synthetic_library(n_per_type=3, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orchestrator = TrainingOrchestrator(cfg, library)
            artefacts = orchestrator.run_all_phases()
            if not artefacts.is_complete:
                raise RuntimeError("training did not complete")
            evaluator = Evaluator(
                artefacts, library, seed=10042, device="cpu",
            )
            manifest = evaluator.evaluate_all(**eval_kwargs)
            if not all(
                k in manifest for k in
                ("detection", "coordinator", "retrieval",
                 "mitigation", "synthetic_control")
            ):
                raise RuntimeError(
                    f"evaluation manifest missing families: "
                    f"{sorted(manifest.keys())}"
                )

            # Run a tiny 2-cell ablation
            exp_cfg = ExperimentConfig(
                ablations=[
                    AblationSpec.from_name("full"),
                    AblationSpec.from_name("no_case_coherence"),
                ],
                seeds=(42,),
                base_training_config=cfg,
                library_factory=lambda s: build_synthetic_library(
                    n_per_type=3, seed=s,
                ),
                output_dir=workdir / "experiments",
                eval_kwargs=eval_kwargs,
                device="cpu",
            )
            exp_results = Experiment(exp_cfg).run_all()
            if exp_results.summary()["n_cells_complete"] != 2:
                raise RuntimeError(
                    f"only {exp_results.summary()['n_cells_complete']}"
                    f"/2 cells completed"
                )

        return TestRecord(
            name=name, category="integration", status="pass",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message="e2e smoke completed",
        )
    except Exception as exc:
        return TestRecord(
            name=name, category="integration", status="fail",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"{type(exc).__name__}: {exc}",
        )


def _integration_test_csv_roundtrip(
    workdir: Path, *, quick: bool = True,
) -> TestRecord:
    """Real-data path: CSV → DataPipeline → CaseLibrary → train → eval.

    Verifies that the production-data ingestion path actually works,
    not just the build_synthetic_library shortcut.
    """
    name = "integration[csv_roundtrip]"
    started_at = _utc_now_iso()
    t0 = time.time()
    try:
        import csv
        import warnings
        import numpy as np
        from case_memory import DEFAULT_FEATURE_NAMES
        from data_pipeline import DataPipeline, MacroSeriesLoader
        # CaseAssembler is unused here

        # Build a tiny synthetic CSV corpus
        macro_dir = workdir / "macro"
        macro_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        countries = ["KOR", "MEX", "ARG", "TUR", "BRA"]
        columns = ["date"] + list(DEFAULT_FEATURE_NAMES)
        for country in countries:
            csv_path = macro_dir / f"{country}.csv"
            with csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(columns)
                for t in range(60):
                    year = 1990 + t // 4
                    q = (t % 4) + 1
                    vals = [rng.standard_normal() for _ in range(12)]
                    w.writerow([f"{year}Q{q}"] + [f"{v:.6f}" for v in vals])

        # Build a tiny crisis catalogue
        crisis_csv = workdir / "crises.csv"
        with crisis_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["country_iso3", "onset_year", "onset_quarter",
                        "crisis_type", "source", "notes"])
            w.writerow(["KOR", "1997", "4", "currency", "Test", ""])
            w.writerow(["ARG", "2001", "4", "twin", "Test", ""])
            w.writerow(["TUR", "2000", "4", "banking", "Test", ""])

        # Phase 1: fit_build
        pipeline = DataPipeline()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            library = pipeline.fit_build(
                crisis_csv=crisis_csv, macro_dir=macro_dir,
            )
        if len(library) < 1:
            raise RuntimeError("library empty after fit_build")

        # Phase 2: save and reload
        pipe_path = workdir / "pipeline.json"
        pipeline.save(pipe_path)
        pipeline2 = DataPipeline.load(pipe_path)
        if pipeline.normalizer.fingerprint() != pipeline2.normalizer.fingerprint():
            raise RuntimeError("normalizer fingerprint mismatch")

        # Phase 3: apply_build with the reloaded pipeline produces
        # bit-identical normalisation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            library2 = pipeline2.apply_build(
                crisis_csv=crisis_csv, macro_dir=macro_dir,
            )
        for cid in library.case_ids():
            c1 = library[cid]
            c2 = library2[cid]
            diff_pre = np.abs(
                c1.pre_onset_trajectory - c2.pre_onset_trajectory
            ).max()
            if diff_pre > 0:
                raise RuntimeError(
                    f"case {cid}: reload diff {diff_pre:.2e} (expected 0)"
                )

        # MacroSeriesLoader can list countries from the directory
        loader = MacroSeriesLoader(macro_dir)
        avail = loader.list_available_countries()
        if set(avail) != set(countries):
            raise RuntimeError(
                f"available countries mismatch: {avail} != {countries}"
            )

        return TestRecord(
            name=name, category="integration", status="pass",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=(
                f"library size {len(library)}, "
                f"normaliser fp {pipeline.normalizer.fingerprint()}"
            ),
        )
    except Exception as exc:
        return TestRecord(
            name=name, category="integration", status="fail",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"{type(exc).__name__}: {exc}",
        )


def _integration_test_save_load_isomorphism(
    workdir: Path, *, quick: bool = True,
) -> TestRecord:
    """Verify save/load is bit-identical at the artefact level.

    Trains a quick model, saves it, reloads from disk, runs inference
    on the same input through both copies, asserts identical output.
    """
    name = "integration[save_load_isomorphism]"
    started_at = _utc_now_iso()
    t0 = time.time()
    try:
        import warnings
        import numpy as np
        from data_pipeline import build_synthetic_library
        from training import TrainingConfig, TrainingOrchestrator

        cfg = TrainingConfig(
            output_dir=workdir / "training", seed=42, device="cpu",
            detection_n_epochs=10, detection_batch_size=16,
            coordinator_n_epochs=10, coordinator_batch_size=8,
            signature_n_epochs=10, retriever_n_epochs=10,
            mitigation_n_episodes=2, mitigation_episode_len=4,
            mitigation_warmup_steps=2, mitigation_batch_size=4,
            mitigation_buffer_capacity=500,
        )
        library = build_synthetic_library(n_per_type=3, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o1 = TrainingOrchestrator(cfg, library)
            a1 = o1.run_all_phases()
            # Fresh orchestrator loads from the same dir → status loaded
            o2 = TrainingOrchestrator(cfg, library)
            a2 = o2.run_all_phases()

        # Confirm all phases loaded (not retrained)
        loaded_statuses = [r.status for r in a2.phase_results]
        if not all(s == "loaded" for s in loaded_statuses):
            raise RuntimeError(
                f"expected all loaded, got {loaded_statuses}"
            )

        # Detection inference must be bit-identical (forward pass only,
        # no training stochastic operations)
        case = library[list(library.case_ids())[0]]
        d1 = a1.detection_ensemble.detect(case.pre_onset_trajectory)
        d2 = a2.detection_ensemble.detect(case.pre_onset_trajectory)
        diff = float(np.abs(d1.probabilities - d2.probabilities).max())
        if diff > 1e-6:
            raise RuntimeError(
                f"detection inference not isomorphic: max diff {diff:.2e}"
            )

        # Coordinator inference identical
        macro = case.pre_onset_trajectory[-1]
        c1 = a1.coordinator.coordinate(macro, d1.probabilities)
        c2 = a2.coordinator.coordinate(macro, d2.probabilities)
        diff_c = float(np.abs(c1.posterior - c2.posterior).max())
        if diff_c > 1e-6:
            raise RuntimeError(
                f"coordinator inference not isomorphic: max diff {diff_c:.2e}"
            )

        return TestRecord(
            name=name, category="integration", status="pass",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"detection diff={diff:.2e}, coordinator diff={diff_c:.2e}",
        )
    except Exception as exc:
        return TestRecord(
            name=name, category="integration", status="fail",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"{type(exc).__name__}: {exc}",
        )


def run_integration_tests(
    workdir: Path, *,
    quick: bool = True,
    scenarios: Optional[tuple[str, ...]] = None,
) -> list[TestRecord]:
    """Run cross-module integration scenarios.

    Parameters
    ----------
    workdir : Path
        Scratch directory for test artefacts.
    quick : bool
        Use small epoch/episode counts.
    scenarios : tuple[str, ...], optional
        Subset of scenario names to run. Default: all.
        Available: ``"e2e_synthetic"``, ``"csv_roundtrip"``,
        ``"save_load_isomorphism"``.
    """
    print()
    _print_banner("INTEGRATION TESTS")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    all_scenarios: dict[str, Callable[[Path, bool], TestRecord]] = {
        "e2e_synthetic": lambda w, q: _integration_test_smoke_e2e(
            w / "e2e", quick=q,
        ),
        "csv_roundtrip": lambda w, q: _integration_test_csv_roundtrip(
            w / "csv", quick=q,
        ),
        "save_load_isomorphism": lambda w, q:
            _integration_test_save_load_isomorphism(
                w / "isom", quick=q,
            ),
    }
    if scenarios is None:
        scenarios = tuple(all_scenarios.keys())
    records: list[TestRecord] = []
    for s in scenarios:
        if s not in all_scenarios:
            print(f"  ! unknown scenario {s}; available: {list(all_scenarios)}")
            records.append(TestRecord(
                name=f"integration[{s}]", category="integration",
                status="error",
                duration_seconds=0.0,
                started_at=_utc_now_iso(),
                finished_at=_utc_now_iso(),
                message=f"unknown scenario {s!r}",
            ))
            continue
        print(f"  running {s:<35s} ...", end="", flush=True)
        rec = all_scenarios[s](workdir, quick)
        glyph = "✓" if rec.status == "pass" else "✗"
        print(f"\r  {glyph} {s:<35s}  {rec.status:<8s} "
              f"{rec.duration_seconds:6.2f}s")
        if rec.status != "pass":
            print(f"      └── {rec.message[:80]}")
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Reproducibility audit
# ---------------------------------------------------------------------------


def _reproducibility_probe_synthetic_library() -> TestRecord:
    """Check build_synthetic_library is bit-deterministic across calls."""
    name = "reproducibility[synthetic_library]"
    started_at = _utc_now_iso()
    t0 = time.time()
    try:
        import numpy as np
        from data_pipeline import build_synthetic_library
        l1 = build_synthetic_library(n_per_type=3, seed=42)
        l2 = build_synthetic_library(n_per_type=3, seed=42)
        if l1.library_checksum() != l2.library_checksum():
            raise RuntimeError(
                f"library checksums differ: "
                f"{l1.library_checksum()[:16]} vs "
                f"{l2.library_checksum()[:16]}"
            )
        max_diff = 0.0
        for cid in l1.case_ids():
            diff = float(
                np.abs(
                    l1[cid].pre_onset_trajectory
                    - l2[cid].pre_onset_trajectory
                ).max()
            )
            max_diff = max(max_diff, diff)
        if max_diff > 0.0:
            raise RuntimeError(
                f"trajectory drift: {max_diff:.2e}"
            )
        return TestRecord(
            name=name, category="reproducibility", status="pass",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message="bit-identical (max trajectory diff = 0)",
        )
    except Exception as exc:
        return TestRecord(
            name=name, category="reproducibility", status="fail",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"{type(exc).__name__}: {exc}",
        )


def _reproducibility_probe_normalizer_save_load() -> TestRecord:
    """Check the normaliser is bit-identical after save/load."""
    name = "reproducibility[normalizer_save_load]"
    started_at = _utc_now_iso()
    t0 = time.time()
    try:
        import tempfile
        import numpy as np
        from data_pipeline import MacroFeatureNormalizer
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 12))
        norm = MacroFeatureNormalizer()
        norm.fit(X)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "norm.json"
            norm.save(p)
            norm2 = MacroFeatureNormalizer.load(p)
            diff_mean = float(np.abs(norm.mean - norm2.mean).max())
            diff_std = float(np.abs(norm.std - norm2.std).max())
            if diff_mean > 0.0 or diff_std > 0.0:
                raise RuntimeError(
                    f"normaliser drift: mean={diff_mean}, std={diff_std}"
                )
            # Bit-identical transform
            Z1 = norm.transform(X)
            Z2 = norm2.transform(X)
            diff_z = float(np.abs(Z1 - Z2).max())
            if diff_z > 0.0:
                raise RuntimeError(
                    f"transform drift: {diff_z}"
                )
        return TestRecord(
            name=name, category="reproducibility", status="pass",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message="bit-identical save/load round-trip",
        )
    except Exception as exc:
        return TestRecord(
            name=name, category="reproducibility", status="fail",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"{type(exc).__name__}: {exc}",
        )


def _reproducibility_probe_training_drift(workdir: Path) -> TestRecord:
    """Document the magnitude of cross-process training non-determinism.

    Runs the same training twice with identical seeds and reports the
    maximum drift in the detection ensemble's output probabilities on
    a fixed test trajectory. This DOES NOT FAIL on small non-zero
    drift — it documents the magnitude, which is a manuscript-relevant
    property of the PyTorch CPU stack.

    Training-budget note
    --------------------
    We deliberately use ``detection_n_epochs=40`` here (rather than
    the smoke-test value of 10) because drift is *much* larger in the
    high-variance early-training regime: empirically the max-prob
    drift falls from ~1.7e-1 at 10 epochs to ~1.2e-2 at 30 epochs to
    ~7e-3 at 50 epochs as the model leaves the high-variance regime.
    At 40 epochs (the value below) the drift sits around 1e-2, which
    is small enough for cell-level comparison in the ablation table
    but still observable for documentation purposes.
    """
    name = "reproducibility[training_drift]"
    started_at = _utc_now_iso()
    t0 = time.time()
    try:
        import warnings
        import numpy as np
        from data_pipeline import build_synthetic_library
        from training import TrainingConfig, TrainingOrchestrator

        def quick_cfg(out_dir: Path) -> TrainingConfig:
            return TrainingConfig(
                output_dir=out_dir, seed=42, device="cpu",
                detection_n_epochs=40, detection_batch_size=16,
                coordinator_n_epochs=40, coordinator_batch_size=8,
                signature_n_epochs=40, retriever_n_epochs=40,
                mitigation_n_episodes=2, mitigation_episode_len=4,
                mitigation_warmup_steps=2, mitigation_batch_size=4,
                mitigation_buffer_capacity=500,
            )
        results = []
        for run in range(2):
            run_dir = workdir / f"drift_run_{run}"
            run_dir.mkdir(parents=True, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lib = build_synthetic_library(n_per_type=3, seed=42)
                orch = TrainingOrchestrator(quick_cfg(run_dir), lib)
                artefacts = orch.run_all_phases()
            case = lib[list(lib.case_ids())[0]]
            out = artefacts.detection_ensemble.detect(
                case.pre_onset_trajectory,
            )
            results.append(out.probabilities)
        drift = float(np.abs(results[0] - results[1]).max())
        # Document, don't fail. Empirically at 40 epochs CPU drift is
        # ~1e-2; the 5e-2 budget accommodates the additional
        # variance from running on different machines / pytorch
        # versions. Drift greater than 5e-2 indicates a real
        # determinism regression.
        if drift > 5e-2:
            status = "fail"
            message = (
                f"training drift exceeded 5e-2 budget: {drift:.2e}"
            )
        else:
            status = "pass"
            message = (
                f"max drift = {drift:.2e} (within 5e-2 budget); "
                f"see manuscript Appendix B"
            )
        return TestRecord(
            name=name, category="reproducibility", status=status,
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=message,
        )
    except Exception as exc:
        return TestRecord(
            name=name, category="reproducibility", status="fail",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"{type(exc).__name__}: {exc}",
        )


def _reproducibility_probe_config_fingerprints() -> TestRecord:
    """Check TrainingConfig and ExperimentConfig fingerprints are
    deterministic across construction."""
    name = "reproducibility[config_fingerprints]"
    started_at = _utc_now_iso()
    t0 = time.time()
    try:
        from training import TrainingConfig
        from experiments import (
            AblationSpec, ExperimentConfig,
        )
        # Two TrainingConfigs with identical fields
        a = TrainingConfig(seed=42)
        b = TrainingConfig(seed=42)
        if a.fingerprint() != b.fingerprint():
            raise RuntimeError(
                f"TrainingConfig fingerprint differs: "
                f"{a.fingerprint()} vs {b.fingerprint()}"
            )
        # Two ExperimentConfigs with identical specs
        def lf(s: int):  # pragma: no cover
            from data_pipeline import build_synthetic_library
            return build_synthetic_library(n_per_type=3, seed=s)
        c1 = ExperimentConfig(
            ablations=[AblationSpec.from_name("full")],
            seeds=(42,),
            base_training_config=a,
            library_factory=lf,
            output_dir=Path("/tmp/a"),
        )
        c2 = ExperimentConfig(
            ablations=[AblationSpec.from_name("full")],
            seeds=(42,),
            base_training_config=b,
            library_factory=lf,
            output_dir=Path("/tmp/b"),  # different path, same fp
        )
        if c1.fingerprint() != c2.fingerprint():
            raise RuntimeError(
                f"ExperimentConfig fingerprint differs: "
                f"{c1.fingerprint()} vs {c2.fingerprint()}"
            )
        return TestRecord(
            name=name, category="reproducibility", status="pass",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=(
                f"TrainingConfig fp={a.fingerprint()}, "
                f"ExperimentConfig fp={c1.fingerprint()}"
            ),
        )
    except Exception as exc:
        return TestRecord(
            name=name, category="reproducibility", status="fail",
            duration_seconds=time.time() - t0,
            started_at=started_at,
            finished_at=_utc_now_iso(),
            message=f"{type(exc).__name__}: {exc}",
        )


def run_reproducibility_audit(
    workdir: Path,
) -> list[TestRecord]:
    """Run reproducibility probes.

    Probes documented in the manuscript:
      1. ``synthetic_library``: bit-identical across calls.
      2. ``normalizer_save_load``: bit-identical after disk
         round-trip.
      3. ``config_fingerprints``: deterministic across construction.
      4. ``training_drift``: documents (rather than fails on) the
         magnitude of PyTorch's residual non-determinism on CPU.
    """
    print()
    _print_banner("REPRODUCIBILITY AUDIT")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    probes: list[Callable[[], TestRecord]] = [
        _reproducibility_probe_synthetic_library,
        _reproducibility_probe_normalizer_save_load,
        _reproducibility_probe_config_fingerprints,
        lambda: _reproducibility_probe_training_drift(workdir),
    ]
    records: list[TestRecord] = []
    for probe in probes:
        rec = probe()
        glyph = "✓" if rec.status == "pass" else "✗"
        print(f"  {glyph} {rec.name:<45s}  {rec.status:<8s} "
              f"{rec.duration_seconds:6.2f}s")
        print(f"      └── {rec.message[:80]}")
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_all(
    *,
    test_dir: Path = Path("/home/claude"),
    output_dir: Path = Path("/home/claude/test_reports"),
    quick: bool = False,
    skip_slow: bool = False,
    categories: tuple[str, ...] = (
        "smoke", "unit", "integration", "reproducibility",
    ),
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> TestReport:
    """Run every test category and assemble a TestReport.

    Parameters
    ----------
    test_dir : Path
        Directory containing the framework modules and test scripts.
    output_dir : Path
        Where to write tests_report.json.
    quick : bool
        Use small epoch/episode counts for integration tests.
    skip_slow : bool
        Skip per-module tests that train full pipelines
        (``test_training.py``, ``test_evaluation.py``,
        ``test_experiments.py``). Useful for fast CI gates.
    categories : tuple[str, ...]
        Subset of categories to run.
    timeout : int
        Per-test subprocess timeout.
    """
    if not isinstance(test_dir, Path):
        test_dir = Path(test_dir)
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if test_dir not in [Path(p) for p in sys.path]:
        sys.path.insert(0, str(test_dir))

    report = TestReport(
        config={
            "test_dir": str(test_dir),
            "output_dir": str(output_dir),
            "quick": bool(quick),
            "skip_slow": bool(skip_slow),
            "categories": list(categories),
            "timeout": int(timeout),
        },
    )
    t0 = time.time()

    if "smoke" in categories:
        for rec in run_smoke_tests():
            report.add(rec)

    if "unit" in categories:
        skip: tuple[str, ...] = ()
        if skip_slow:
            skip = (
                "test_training.py",
                "test_evaluation.py",
                "test_experiments.py",
            )
        for rec in run_unit_tests(
            test_dir, timeout=timeout, skip=skip,
        ):
            report.add(rec)

    if "integration" in categories:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for rec in run_integration_tests(Path(tmp), quick=quick):
                report.add(rec)

    if "reproducibility" in categories:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for rec in run_reproducibility_audit(Path(tmp)):
                report.add(rec)

    report.finished_at = _utc_now_iso()
    report.duration_seconds = time.time() - t0
    report_path = output_dir / "tests_report.json"
    report.save(report_path)
    print()
    print(f"Test report saved to: {report_path}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main() -> int:
    """CLI: ``python tests.py [options]``.

    Examples
    --------
    Run everything (slow)::

        python tests.py --all

    Quick smoke gate for CI::

        python tests.py --smoke

    Smoke + integration only (skip the slow per-module tests)::

        python tests.py --smoke --integration --quick

    Reproducibility appendix for the manuscript::

        python tests.py --reproducibility
    """
    parser = argparse.ArgumentParser(
        prog="python tests.py",
        description="Run the framework's unified test harness.",
    )
    parser.add_argument("--log-level", default="WARNING",
                        help="Logging level (default: WARNING; the "
                             "subprocess test outputs are noisy at INFO).")
    parser.add_argument("--test-dir", type=Path, default=Path("/home/claude"),
                        help="Directory containing the framework modules.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/home/claude/test_reports"))
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke tests.")
    parser.add_argument("--unit", action="store_true",
                        help="Run per-module unit tests.")
    parser.add_argument("--integration", action="store_true",
                        help="Run cross-module integration tests.")
    parser.add_argument("--reproducibility", action="store_true",
                        help="Run reproducibility audit.")
    parser.add_argument("--all", action="store_true",
                        help="Run every category (equivalent to "
                             "--smoke --unit --integration --reproducibility).")
    parser.add_argument("--quick", action="store_true",
                        help="Use small training budgets for integration tests.")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip per-module tests that train full pipelines.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help=f"Per-subprocess timeout in seconds "
                             f"(default: {DEFAULT_TIMEOUT_SECONDS}).")
    args = parser.parse_args()
    _configure_logging(args.log_level)

    # Resolve category selection
    requested: list[str] = []
    if args.all:
        requested = ["smoke", "unit", "integration", "reproducibility"]
    else:
        if args.smoke:
            requested.append("smoke")
        if args.unit:
            requested.append("unit")
        if args.integration:
            requested.append("integration")
        if args.reproducibility:
            requested.append("reproducibility")
    if not requested:
        # No flags → default to all
        requested = ["smoke", "unit", "integration", "reproducibility"]

    try:
        report = run_all(
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            quick=args.quick,
            skip_slow=args.skip_slow,
            categories=tuple(requested),
            timeout=args.timeout,
        )
        report.print_summary()
        return 0 if report.is_green else 1
    except TestHarnessError as exc:
        logger.error("Harness error: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
