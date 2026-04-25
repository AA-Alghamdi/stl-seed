"""Phase-2 progress monitor for the canonical 18-cell sweep.

Run this LOCALLY on the M5 Pro while the sweep executes on a remote
RunPod 4090 spot pod. It surfaces per-cell status (PENDING / RUNNING /
COMPLETED / FAILED), an estimated time-remaining curve from the
per-model wall-clock priors in ``configs/model/qwen3_*.yaml``, and a
running cost burn-down against the registered $25 cap.

Two backends, in priority order:

  1. **HuggingFace Hub** — when the sweep was launched with
     ``hf_hub.enabled=true``, each finished cell pushes a LoRA adapter
     to ``<repo_org>/<repo_prefix>-<cell_id>``. This monitor lists the
     org's repos that match the prefix and treats the presence of an
     ``adapter_config.json`` file in a repo as the COMPLETED signal.
     Survives spot-interruption: HF Hub is the only ground truth a
     local M5 Pro can poll without an SSH tunnel into the pod.

  2. **Local file system** — falls back to ``runs/canonical/sweep_log.csv``
     and ``runs/canonical/<cell_id>/done.flag`` (written by
     ``scripts/run_canonical_sweep.py``). This is the path during early
     Phase 2 when HF push is still disabled, *or* when the user is
     monitoring from inside the pod itself.

Both modes refresh on a configurable interval (default 60 s). The
``--once`` flag does a single render and exits, suitable for cron.


Usage::

    # Local poll of an in-progress sweep that is pushing to HF Hub:
    uv run python scripts/sweep_monitor.py \\
        --hf-org AA-Alghamdi --hf-prefix stl-seed-phase2

    # Local poll using only the on-disk sweep log (no HF push):
    uv run python scripts/sweep_monitor.py --runs-dir runs/canonical

    # One-shot for cron / CI dashboards:
    uv run python scripts/sweep_monitor.py --hf-org AA-Alghamdi --once
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Paths and constants. Mirror run_canonical_sweep.py so the two stay in
# lockstep without importing from it (lazy-import discipline preserves
# `--help` on M5 Pro without diffrax/jax/torch installed).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RUNS_DIR = _REPO_ROOT / "runs" / "canonical"
_SWEEP_LOG_NAME = "sweep_log.csv"

# Per-model expected wall-clock minutes. Source: configs/model/qwen3_*.yaml
# (`expected_minutes_per_cell`). Inlined here to keep this script free of
# Hydra / OmegaConf imports — the monitor must work on a stripped clone
# without the heavy dependency stack.
_MODEL_MINUTES: dict[str, int] = {
    "qwen3_0.6b": 18,
    "qwen3_1.7b": 28,
    "qwen3_4b": 55,
}

# Canonical sweep enumeration. Mirrors configs/sweep_main.yaml exactly.
# Locked here so the monitor can render its dashboard when neither HF
# nor the local file system have any cells yet (PENDING-everything
# initial state).
_CANONICAL_MODELS = ("qwen3_0.6b", "qwen3_1.7b", "qwen3_4b")
_CANONICAL_FILTERS = ("hard", "quantile", "continuous")
_CANONICAL_TASKS = ("bio_ode_repressilator", "glucose_insulin")

# Default per-hour GPU rate. RunPod 4090 spot at 2026-04 pricing per
# docker/runpod_README.md §0. Override via --gpu-dollars-per-hour for
# A6000 ($0.49) or A100 ($1.19) fallbacks.
_DEFAULT_GPU_DOLLARS_PER_HOUR = 0.34
_DEFAULT_BUDGET_USD = 25.0

# Cell-status string set; kept in one place so the dashboard renderer
# and the aggregator agree on capitalization.
_STATUS_PENDING = "PENDING"
_STATUS_RUNNING = "RUNNING"
_STATUS_COMPLETED = "COMPLETED"
_STATUS_FAILED = "FAILED"
_STATUS_INTERRUPTED = "INTERRUPTED"
_STATUS_SKIPPED = "SKIPPED"

# Map sweep_log.csv `status` values onto the monitor's status set so the
# two surfaces are consistent. The sweep writes "OK" on success; we
# render that as COMPLETED.
_LOG_STATUS_MAP: dict[str, str] = {
    "OK": _STATUS_COMPLETED,
    "DRY_RUN": _STATUS_COMPLETED,
    "FAILED": _STATUS_FAILED,
    "SKIPPED": _STATUS_SKIPPED,
    "INTERRUPTED": _STATUS_INTERRUPTED,
}

console = Console()


# ---------------------------------------------------------------------------
# Cell + status dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One (model, filter, task) cell of the canonical sweep."""

    model: str
    filter: str
    task: str

    @property
    def cell_id(self) -> str:
        return f"{self.model}__{self.filter}__{self.task}"


@dataclass
class CellStatus:
    """Snapshot of one cell's current status, observed via HF or FS."""

    cell: Cell
    status: str = _STATUS_PENDING
    duration_s: float = 0.0
    cost_usd: float = 0.0
    source: str = "none"  # "hf" | "fs" | "none"
    note: str = ""


@dataclass
class SweepSnapshot:
    """Aggregate status across all cells at one polling instant."""

    cells: list[CellStatus]
    backend: str  # "hf" | "fs" | "merged"
    polled_at: float = field(default_factory=time.time)

    def by_status(self, status: str) -> list[CellStatus]:
        return [c for c in self.cells if c.status == status]

    @property
    def n_total(self) -> int:
        return len(self.cells)

    @property
    def n_completed(self) -> int:
        return len(self.by_status(_STATUS_COMPLETED))

    @property
    def n_failed(self) -> int:
        return len(self.by_status(_STATUS_FAILED))

    @property
    def n_running(self) -> int:
        return len(self.by_status(_STATUS_RUNNING))

    @property
    def n_pending(self) -> int:
        return len(self.by_status(_STATUS_PENDING))

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.cells)


# ---------------------------------------------------------------------------
# Cell enumeration. Identical to run_canonical_sweep.enumerate_cells.
# ---------------------------------------------------------------------------


def enumerate_canonical_cells() -> list[Cell]:
    """Return the locked 18 cells (3 models x 3 filters x 2 tasks).

    Ordering matches run_canonical_sweep.enumerate_cells: model outermost,
    filter middle, task innermost. Matters because the running-cost
    progression is read off this order.
    """
    return [
        Cell(model=m, filter=f, task=t)
        for m in _CANONICAL_MODELS
        for f in _CANONICAL_FILTERS
        for t in _CANONICAL_TASKS
    ]


# ---------------------------------------------------------------------------
# Cost helpers.
# ---------------------------------------------------------------------------


def expected_minutes(cell: Cell) -> float:
    """Return per-cell wall-clock minutes from the per-model prior."""
    return float(_MODEL_MINUTES.get(cell.model, 30))


def expected_dollars(cell: Cell, dollars_per_hour: float) -> float:
    """Return per-cell expected dollars at ``dollars_per_hour`` GPU rate."""
    return expected_minutes(cell) / 60.0 * dollars_per_hour


# ---------------------------------------------------------------------------
# Local file-system poller.
# ---------------------------------------------------------------------------


def _read_sweep_log(runs_dir: Path) -> dict[str, dict[str, str]]:
    """Read runs/canonical/sweep_log.csv into ``{cell_id: row}``.

    Returns an empty dict if the log does not exist yet (sweep hasn't
    started). Robust to a partially-written tail row (csv.DictReader
    silently drops malformed rows mid-stream — we accept that to avoid
    crashing the dashboard during a live append).
    """
    log_path = runs_dir / _SWEEP_LOG_NAME
    if not log_path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        with log_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("cell_id")
                if cid:
                    # Last-write-wins: a cell may appear more than once
                    # if it was retried; the most recent row is the
                    # ground truth.
                    out[cid] = row
    except (OSError, csv.Error):
        # Sweep is mid-write; return what we have.
        return out
    return out


def _detect_running_cells(runs_dir: Path, completed: set[str]) -> set[str]:
    """A cell is RUNNING if its output dir exists but no done.flag yet.

    The sweep writes provenance.json on entry and done.flag on success.
    A cell with provenance.json but no done.flag and no FAILED row in
    the log is in flight. ``completed`` short-circuits the check for
    cells we already know are finished.
    """
    if not runs_dir.exists():
        return set()
    running: set[str] = set()
    for sub in runs_dir.iterdir():
        if not sub.is_dir():
            continue
        cid = sub.name
        if "__" not in cid or cid in completed:
            continue
        if (sub / "provenance.json").exists() and not (sub / "done.flag").exists():
            running.add(cid)
    return running


def poll_local_filesystem(runs_dir: Path, dollars_per_hour: float) -> SweepSnapshot:
    """Read ``runs/canonical/`` and produce a SweepSnapshot.

    Status precedence (highest to lowest):
      1. ``done.flag`` exists                  -> COMPLETED
      2. sweep_log row says FAILED/INT/SKIP    -> matching status
      3. cell dir exists with provenance.json  -> RUNNING
      4. nothing on disk                       -> PENDING
    """
    cells = enumerate_canonical_cells()
    log_rows = _read_sweep_log(runs_dir)
    completed: set[str] = set()
    statuses: dict[str, CellStatus] = {}

    for cell in cells:
        cid = cell.cell_id
        cell_dir = runs_dir / cid
        done_flag = cell_dir / "done.flag"
        s = CellStatus(cell=cell, source="fs")
        if done_flag.exists():
            s.status = _STATUS_COMPLETED
            completed.add(cid)
            row = log_rows.get(cid, {})
            try:
                s.duration_s = float(row.get("duration_s", 0.0))
                s.cost_usd = float(row.get("cost_usd", 0.0))
            except ValueError:
                pass
            if s.cost_usd <= 0.0:
                # Estimate after-the-fact if the log entry was lost.
                s.cost_usd = expected_dollars(cell, dollars_per_hour)
        elif cid in log_rows:
            row = log_rows[cid]
            mapped = _LOG_STATUS_MAP.get(row.get("status", ""), _STATUS_RUNNING)
            s.status = mapped
            try:
                s.duration_s = float(row.get("duration_s", 0.0))
                s.cost_usd = float(row.get("cost_usd", 0.0))
            except ValueError:
                pass
            if mapped == _STATUS_FAILED:
                s.note = row.get("error", "")[:60]
        statuses[cid] = s

    # Second pass: tag RUNNING cells from on-disk evidence (provenance
    # exists but done.flag does not, and no log row marked it terminal).
    running = _detect_running_cells(runs_dir, completed)
    for cid in running:
        if statuses[cid].status == _STATUS_PENDING:
            statuses[cid].status = _STATUS_RUNNING

    return SweepSnapshot(cells=list(statuses.values()), backend="fs")


# ---------------------------------------------------------------------------
# HuggingFace Hub poller.
# ---------------------------------------------------------------------------


def poll_hf_hub(
    repo_org: str,
    repo_prefix: str,
    dollars_per_hour: float,
) -> SweepSnapshot:
    """Poll HF Hub for ``<repo_org>/<repo_prefix>-<cell_id>`` repos.

    A cell is COMPLETED when its repo exists AND contains an
    ``adapter_config.json`` file (the canonical PEFT marker). Repos
    that exist but have no adapter file yet are RUNNING (push in
    flight or sweep wrote the repo metadata before the adapter
    upload). Missing repos are PENDING.

    Uses the public read-only HfApi surface; no token required for
    public repos. Lazy-imports huggingface_hub so the script's
    ``--help`` works on a stripped clone.
    """
    from huggingface_hub import HfApi  # lazy import
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    cells = enumerate_canonical_cells()
    statuses: list[CellStatus] = []

    # One list_repo_files call per cell; cheap (HEAD-style) and avoids
    # paginating the org's full model index (which may include
    # unrelated repos). For 18 cells this is ~18 round trips per poll;
    # at the default 60 s refresh that's 1080 calls/hour, well under HF
    # Hub's anonymous rate limits.
    for cell in cells:
        cid = cell.cell_id
        repo_id = f"{repo_org}/{repo_prefix}-{cid}"
        s = CellStatus(cell=cell, source="hf")
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            has_adapter = any(
                f.endswith("adapter_config.json") or f.endswith("adapter_model.safetensors")
                for f in files
            )
            if has_adapter:
                s.status = _STATUS_COMPLETED
                # No reliable wall-clock from HF — estimate from the prior.
                s.cost_usd = expected_dollars(cell, dollars_per_hour)
                s.note = repo_id
            else:
                s.status = _STATUS_RUNNING
                s.note = repo_id
        except HfHubHTTPError as exc:
            # 404 = repo doesn't exist yet -> PENDING.
            # Anything else (5xx, rate-limit) -> still PENDING but note it.
            if "404" in str(exc):
                s.status = _STATUS_PENDING
            else:
                s.status = _STATUS_PENDING
                s.note = f"hf err: {type(exc).__name__}"
        except (OSError, ValueError) as exc:
            s.status = _STATUS_PENDING
            s.note = f"net err: {type(exc).__name__}"
        statuses.append(s)

    return SweepSnapshot(cells=statuses, backend="hf")


def merge_snapshots(hf: SweepSnapshot, fs: SweepSnapshot) -> SweepSnapshot:
    """Combine HF and FS snapshots; HF COMPLETED is authoritative.

    Rule: a cell is COMPLETED if either backend says so. RUNNING/FAILED
    from FS take precedence over HF PENDING (the local pod has more
    detail than the remote artifact store mid-sweep).
    """
    by_cid: dict[str, CellStatus] = {c.cell.cell_id: c for c in fs.cells}
    for h in hf.cells:
        cid = h.cell.cell_id
        cur = by_cid.get(cid)
        if cur is None:
            by_cid[cid] = h
            continue
        if h.status == _STATUS_COMPLETED:
            # Promote FS to COMPLETED if HF says it's done.
            cur.status = _STATUS_COMPLETED
            cur.source = "hf+fs"
            if cur.cost_usd <= 0.0:
                cur.cost_usd = h.cost_usd
            if not cur.note:
                cur.note = h.note
        elif cur.status == _STATUS_PENDING and h.status == _STATUS_RUNNING:
            cur.status = _STATUS_RUNNING
            cur.source = "hf"
            cur.note = h.note
    return SweepSnapshot(cells=list(by_cid.values()), backend="merged")


# ---------------------------------------------------------------------------
# Time-remaining estimator.
# ---------------------------------------------------------------------------


def estimate_remaining(snap: SweepSnapshot, dollars_per_hour: float) -> dict[str, float]:
    """Compute (minutes_remaining, dollars_remaining, pct_complete).

    Uses per-cell wall-clock priors for PENDING + RUNNING cells. RUNNING
    cells are given half their expected time as a midpoint estimate
    (we don't know how far into them we are without per-step heartbeats).
    """
    pending_minutes = 0.0
    pending_dollars = 0.0
    for cs in snap.cells:
        if cs.status == _STATUS_PENDING:
            pending_minutes += expected_minutes(cs.cell)
            pending_dollars += expected_dollars(cs.cell, dollars_per_hour)
        elif cs.status == _STATUS_RUNNING:
            pending_minutes += expected_minutes(cs.cell) * 0.5
            pending_dollars += expected_dollars(cs.cell, dollars_per_hour) * 0.5
    pct = 100.0 * snap.n_completed / max(snap.n_total, 1)
    return {
        "minutes_remaining": pending_minutes,
        "hours_remaining": pending_minutes / 60.0,
        "dollars_remaining": pending_dollars,
        "pct_complete": pct,
    }


# ---------------------------------------------------------------------------
# Dashboard rendering.
# ---------------------------------------------------------------------------


_STATUS_STYLE: dict[str, str] = {
    _STATUS_COMPLETED: "[green]COMPLETED[/]",
    _STATUS_RUNNING: "[yellow]RUNNING[/]",
    _STATUS_FAILED: "[red]FAILED[/]",
    _STATUS_PENDING: "[dim]PENDING[/]",
    _STATUS_INTERRUPTED: "[magenta]INTERRUPTED[/]",
    _STATUS_SKIPPED: "[blue]SKIPPED[/]",
}


def render_dashboard(
    snap: SweepSnapshot,
    *,
    budget_usd: float,
    dollars_per_hour: float,
    backend_label: str,
) -> Panel:
    """Render the full dashboard (header + cell table) as a Rich Panel."""
    rem = estimate_remaining(snap, dollars_per_hour)
    spent = snap.total_cost_usd
    burn_color = (
        "green" if spent <= budget_usd * 0.6 else "yellow" if spent <= budget_usd else "red"
    )

    header = (
        f"[bold]stl-seed Phase-2 sweep monitor[/]\n"
        f"  backend: {backend_label}\n"
        f"  polled at: {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(snap.polled_at))}\n"
        f"  cells: [green]{snap.n_completed}[/] done  "
        f"[yellow]{snap.n_running}[/] running  "
        f"[red]{snap.n_failed}[/] failed  "
        f"[dim]{snap.n_pending}[/] pending  "
        f"(/ {snap.n_total} total)\n"
        f"  progress: {rem['pct_complete']:.1f}%\n"
        f"  est remaining: {rem['hours_remaining']:.1f} h  "
        f"~${rem['dollars_remaining']:.2f}\n"
        f"  cost burn: [{burn_color}]${spent:.2f}[/] / ${budget_usd:.2f} cap  "
        f"(@ ${dollars_per_hour:.2f}/h)"
    )

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("cell_id")
    table.add_column("status", width=12)
    table.add_column("source", width=8)
    table.add_column("dur (s)", justify="right", width=8)
    table.add_column("cost ($)", justify="right", width=8)
    table.add_column("note", overflow="ellipsis")
    for i, cs in enumerate(snap.cells, start=1):
        table.add_row(
            str(i),
            cs.cell.cell_id,
            _STATUS_STYLE.get(cs.status, cs.status),
            cs.source,
            f"{cs.duration_s:.0f}" if cs.duration_s > 0 else "-",
            f"{cs.cost_usd:.2f}" if cs.cost_usd > 0 else "-",
            cs.note or "",
        )

    from rich.console import Group  # local import: keeps top-of-file deps minimal

    return Panel(Group(header, table), title="canonical sweep monitor", border_style="cyan")


# ---------------------------------------------------------------------------
# Top-level orchestration.
# ---------------------------------------------------------------------------


def take_snapshot(
    *,
    hf_org: str | None,
    hf_prefix: str,
    runs_dir: Path,
    dollars_per_hour: float,
) -> SweepSnapshot:
    """Choose backend(s) and produce a single snapshot.

    Both backends run when ``hf_org`` is set AND the runs dir exists;
    the merged view wins because each backend has different blind
    spots. HF-only is the typical local-monitor case; FS-only is the
    in-pod case.
    """
    hf_snap: SweepSnapshot | None = None
    fs_snap: SweepSnapshot | None = None

    if hf_org is not None:
        try:
            hf_snap = poll_hf_hub(hf_org, hf_prefix, dollars_per_hour)
        except ImportError:
            console.print("[yellow]huggingface_hub not installed; skipping HF backend[/]")
            hf_snap = None
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]HF poll failed: {type(exc).__name__}: {exc}[/]")
            hf_snap = None

    if runs_dir.exists():
        try:
            fs_snap = poll_local_filesystem(runs_dir, dollars_per_hour)
        except (OSError, ValueError) as exc:
            console.print(f"[yellow]FS poll failed: {type(exc).__name__}: {exc}[/]")
            fs_snap = None

    if hf_snap is not None and fs_snap is not None:
        return merge_snapshots(hf_snap, fs_snap)
    if hf_snap is not None:
        return hf_snap
    if fs_snap is not None:
        return fs_snap

    # Nothing observable — return all-PENDING so the dashboard renders
    # the locked 18-cell enumeration anyway.
    return SweepSnapshot(
        cells=[CellStatus(cell=c) for c in enumerate_canonical_cells()],
        backend="none",
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sweep_monitor.py",
        description="Phase-2 progress monitor for the canonical 18-cell sweep.",
    )
    p.add_argument(
        "--hf-org",
        type=str,
        default=None,
        help=(
            "HuggingFace org/user that the sweep pushes to. If unset, the "
            "monitor falls back to the local file system."
        ),
    )
    p.add_argument(
        "--hf-prefix",
        type=str,
        default="stl-seed-phase2",
        help="Repo prefix; full repo id = <org>/<prefix>-<cell_id>. (default: stl-seed-phase2)",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=_DEFAULT_RUNS_DIR,
        help="Local sweep output dir; the FS fallback reads sweep_log.csv from here.",
    )
    p.add_argument(
        "--gpu-dollars-per-hour",
        type=float,
        default=_DEFAULT_GPU_DOLLARS_PER_HOUR,
        help="Per-hour GPU rate for cost estimates (default: 0.34 = 4090 spot).",
    )
    p.add_argument(
        "--budget-usd",
        type=float,
        default=_DEFAULT_BUDGET_USD,
        help="Registered cost cap (default: 25.0). Burn-down is shown vs. this.",
    )
    p.add_argument(
        "--refresh-seconds",
        type=int,
        default=60,
        help="Polling interval in seconds (default: 60).",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Single-shot render, then exit. Useful for cron / status dashboards.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    backend_label = []
    if args.hf_org is not None:
        backend_label.append(f"HF Hub ({args.hf_org}/{args.hf_prefix}-*)")
    if args.runs_dir.exists():
        backend_label.append(f"FS ({args.runs_dir})")
    if not backend_label:
        backend_label.append("none — locked enumeration only")
    label = " + ".join(backend_label)

    if args.once:
        snap = take_snapshot(
            hf_org=args.hf_org,
            hf_prefix=args.hf_prefix,
            runs_dir=args.runs_dir,
            dollars_per_hour=args.gpu_dollars_per_hour,
        )
        console.print(
            render_dashboard(
                snap,
                budget_usd=args.budget_usd,
                dollars_per_hour=args.gpu_dollars_per_hour,
                backend_label=label,
            )
        )
        return 0 if snap.n_failed == 0 else 1

    # Live mode: refresh on the configured interval. Ctrl-C exits cleanly.
    try:
        snap = take_snapshot(
            hf_org=args.hf_org,
            hf_prefix=args.hf_prefix,
            runs_dir=args.runs_dir,
            dollars_per_hour=args.gpu_dollars_per_hour,
        )
        with Live(
            render_dashboard(
                snap,
                budget_usd=args.budget_usd,
                dollars_per_hour=args.gpu_dollars_per_hour,
                backend_label=label,
            ),
            console=console,
            refresh_per_second=2,
            screen=False,
        ) as live:
            while True:
                time.sleep(args.refresh_seconds)
                snap = take_snapshot(
                    hf_org=args.hf_org,
                    hf_prefix=args.hf_prefix,
                    runs_dir=args.runs_dir,
                    dollars_per_hour=args.gpu_dollars_per_hour,
                )
                live.update(
                    render_dashboard(
                        snap,
                        budget_usd=args.budget_usd,
                        dollars_per_hour=args.gpu_dollars_per_hour,
                        backend_label=label,
                    )
                )
                if snap.n_completed == snap.n_total and snap.n_failed == 0:
                    # Sweep finished cleanly; exit so a wrapper script
                    # (notification / Discord webhook) can fire.
                    break
    except KeyboardInterrupt:
        console.print("\n[yellow]monitor stopped (Ctrl-C)[/]")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
