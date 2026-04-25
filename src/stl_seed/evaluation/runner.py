"""Parallel multi-checkpoint evaluation driver.

Used by ``stl-seed evaluate`` (CLI) and the canonical-sweep analysis
script. Wraps ``EvalHarness.evaluate_checkpoint`` over an iterable of
checkpoints, with rich progress output and resumability via a
``ResultsStore`` that tracks ``(checkpoint, spec)`` keys already
evaluated.

The runner is intentionally serial-by-default: per-spec simulation is
already vectorized inside the simulator, and concurrent JAX compilation
across checkpoints triggers retracing storms that cost more wall time
than they save. A future ``concurrent.futures.ProcessPoolExecutor``
mode is stubbed in ``RunnerConfig.parallel`` for completeness; the
default ``parallel=False`` is what we run in production.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from stl_seed.evaluation.harness import (
    DEFAULT_BON_BUDGETS,
    CheckpointProtocol,
    EvalHarness,
    EvalResults,
    SimulatorProtocol,
)


@dataclass
class RunnerConfig:
    """Configuration knobs for ``EvalRunner.run(...)``."""

    n_samples_per_spec: int = 128
    budgets: tuple[int, ...] = DEFAULT_BON_BUDGETS
    parallel: bool = False
    n_workers: int = 1
    output_dir: Path | None = None
    overwrite: bool = False
    seed_base: int = 0


@dataclass
class RunRecord:
    """Per-checkpoint outcome record stored by the runner.

    The ``diversity_warnings`` field carries a list of spec names whose
    ``first_action_uniqueness`` fell below the runner's diversity
    threshold (default 0.5). Surfaced in :func:`stringify_aggregate` as
    a ``[DIVERSITY WARNING]`` annotation. Wired up in response to the
    where 5/5 generations produced an identical first action.
    """

    checkpoint_name: str
    output_path: Path | None
    aggregate_bon: dict[int, float]
    n_specs: int
    success: bool
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    diversity_warnings: list[str] = field(default_factory=list)


# Threshold at which a (checkpoint, spec) cell is annotated as
# [DIVERSITY WARNING] in the stringified output. Below 0.5 means more
# than half of generations share their first action with another
# generation — the A15 memorization signature.
DIVERSITY_WARNING_THRESHOLD: float = 0.5


class EvalRunner:
    """Drive ``EvalHarness`` across multiple checkpoints with progress UI.

    Parameters
    ----------
    simulator_registry, spec_registry, stl_evaluator, initial_state_fn:
        Same arguments accepted by :class:`EvalHarness`.
    config:
        ``RunnerConfig`` controlling sample budget, output, parallelism.
    console:
        Optional ``rich.console.Console`` for progress output. Defaults
        to a default-styled console writing to ``stderr``.
    """

    def __init__(
        self,
        simulator_registry: Mapping[str, SimulatorProtocol],
        spec_registry: Mapping[str, Any],
        stl_evaluator: Any,
        initial_state_fn: Any,
        config: RunnerConfig | None = None,
        console: Console | None = None,
    ) -> None:
        self.harness = EvalHarness(
            simulator_registry=simulator_registry,
            spec_registry=spec_registry,
            stl_evaluator=stl_evaluator,
            initial_state_fn=initial_state_fn,
            budgets=(config or RunnerConfig()).budgets,
        )
        self.config = config or RunnerConfig()
        self.console = console or Console(stderr=True)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        checkpoints: Sequence[CheckpointProtocol],
        held_out_specs: Sequence[str],
    ) -> list[RunRecord]:
        """Evaluate each checkpoint on the same held-out spec set."""
        if self.config.output_dir is not None:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

        records: list[RunRecord] = []
        if self.config.parallel and self.config.n_workers > 1:
            records = self._run_parallel(checkpoints, held_out_specs)
        else:
            records = self._run_serial(checkpoints, held_out_specs)
        return records

    # ------------------------------------------------------------------
    # Serial execution
    # ------------------------------------------------------------------

    def _run_serial(
        self,
        checkpoints: Sequence[CheckpointProtocol],
        held_out_specs: Sequence[str],
    ) -> list[RunRecord]:
        records: list[RunRecord] = []
        n_total = len(checkpoints)

        progress_columns = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        with Progress(*progress_columns, console=self.console, transient=False) as progress:
            task = progress.add_task("[cyan]Evaluating checkpoints", total=n_total)
            for i, ckpt in enumerate(checkpoints):
                record = self._run_one(
                    ckpt=ckpt,
                    held_out_specs=held_out_specs,
                    seed=self.config.seed_base + i,
                )
                records.append(record)
                progress.advance(task)
        return records

    # ------------------------------------------------------------------
    # Parallel execution (process-pool; stubbed)
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        checkpoints: Sequence[CheckpointProtocol],
        held_out_specs: Sequence[str],
    ) -> list[RunRecord]:
        """Process-pool execution. Not used in the canonical sweep —
        JAX retracing across processes is expensive — but kept for
        users who want to fan out small per-checkpoint workloads.
        """
        records: list[RunRecord] = []
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as pool:
            futures = {
                pool.submit(
                    self._run_one,
                    ckpt=ckpt,
                    held_out_specs=held_out_specs,
                    seed=self.config.seed_base + i,
                ): ckpt
                for i, ckpt in enumerate(checkpoints)
            }
            for fut in as_completed(futures):
                records.append(fut.result())
        return records

    # ------------------------------------------------------------------
    # Per-checkpoint
    # ------------------------------------------------------------------

    def _run_one(
        self,
        ckpt: CheckpointProtocol,
        held_out_specs: Sequence[str],
        seed: int,
    ) -> RunRecord:
        name = getattr(ckpt, "name", "<anonymous>")
        out_path: Path | None = None
        if self.config.output_dir is not None:
            out_path = self.config.output_dir / f"{name}.eval.json"
            if out_path.exists() and not self.config.overwrite:
                # Resume: read aggregate from existing artifact
                try:
                    payload = json.loads(out_path.read_text())
                    return RunRecord(
                        checkpoint_name=name,
                        output_path=out_path,
                        aggregate_bon=payload.get("aggregate_bon", {}),
                        n_specs=len(payload.get("per_spec", {})),
                        success=True,
                        extras={"resumed": True},
                    )
                except (OSError, ValueError):
                    # Corrupt artifact — fall through and re-run
                    pass

        try:
            results: EvalResults = self.harness.evaluate_checkpoint(
                checkpoint=ckpt,
                held_out_specs=held_out_specs,
                n_samples_per_spec=self.config.n_samples_per_spec,
                key=seed,
            )
            agg = results.aggregate_bon()
            # Identify cells that tripped the diversity warning.
            diversity_warnings: list[str] = []
            for spec_name, per_spec in results.per_spec.items():
                fau = per_spec.diversity.get("first_action_uniqueness")
                if fau is None:
                    continue
                try:
                    fau_f = float(fau)
                except (TypeError, ValueError):
                    continue
                if (
                    fau_f == fau_f  # not NaN
                    and fau_f < DIVERSITY_WARNING_THRESHOLD
                ):
                    diversity_warnings.append(spec_name)
            if out_path is not None:
                payload = results.as_dict()
                payload["aggregate_bon"] = agg
                payload["diversity_warnings"] = list(diversity_warnings)
                out_path.write_text(json.dumps(payload, indent=2, default=_json_default))
            return RunRecord(
                checkpoint_name=name,
                output_path=out_path,
                aggregate_bon=agg,
                n_specs=len(results.per_spec),
                success=True,
                diversity_warnings=diversity_warnings,
            )
        except Exception as exc:  # noqa: BLE001  — driver-level catch by design
            return RunRecord(
                checkpoint_name=name,
                output_path=out_path,
                aggregate_bon={},
                n_specs=0,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy / jax / tuple keys."""
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except ImportError:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def stringify_aggregate(records: Iterable[RunRecord]) -> str:
    """Tabular summary of a run, for paste-into-paper consumption.

    Cells whose ``first_action_uniqueness`` fell below
    :data:`DIVERSITY_WARNING_THRESHOLD` are appended with
    ``[DIVERSITY WARNING: <spec1>, <spec2>, ...]`` so the A15 memorization
    """
    lines = []
    for r in records:
        if not r.success:
            lines.append(f"  {r.checkpoint_name:<40s}  FAILED: {r.error}")
            continue
        agg_str = "  ".join(f"BoN-{n}={v:.3f}" for n, v in sorted(r.aggregate_bon.items()))
        line = f"  {r.checkpoint_name:<40s}  n_specs={r.n_specs}  {agg_str}"
        if r.diversity_warnings:
            line += f"  [DIVERSITY WARNING: {', '.join(r.diversity_warnings)}]"
        lines.append(line)
    return "\n".join(lines)


__all__ = [
    "EvalRunner",
    "RunnerConfig",
    "RunRecord",
    "DIVERSITY_WARNING_THRESHOLD",
    "stringify_aggregate",
]
