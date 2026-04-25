# RunPod playbook for stl-seed Phase 2

Operator's manual for the canonical training sweep on RunPod. Phase 1
ran on the M5 Pro; Phase 2 is the first time we spend money.

> **TL;DR after bootstrap passes:**
>
> ```bash
> python scripts/run_canonical_sweep.py --config-name sweep_main --max-cost-usd 25 --confirm
> python scripts/run_canonical_eval.py --runs-dir runs/canonical
> # then locally on the M5 Pro:
> python scripts/canonical_analysis.py --eval-results runs/canonical/eval_results.parquet
> ```
>
> First command is gated by `--confirm`; without it, the sweep prints
> the cell enumeration + cost forecast and exits. `--dry-run` is also
> available for local iteration.

## 1. Cost expectations

| Item | Hourly | Hours | Cost |
|---|---|---|---|
| Smoke test on 1×4090 spot | $0.34 | 0.5 | $0.17 |
| Canonical 18-cell sweep (3 sizes × 3 filters × 2 tasks) | $0.34 | ~25 | $8.50 |
| Headroom for re-runs / spot interruptions | — | — | $5 |
| Storage (50 GB × 30 d) | $0.07/GB-mo | — | $3.50 |
| **Suggested initial credit** | | | **$25** |

Pricing is RunPod 4090 spot at ~$0.34/hr (2026-04). Verify on the
dashboard before committing. Fallbacks if 4090 spot is gone: A6000
($0.49/hr spot), A100 40GB ($1.19/hr).

**Hard rule:** never start a pod without setting Pod Termination
(default it to 3 hours; extend manually).

## 2. Sign up

1. Account at <https://runpod.io>; load $25 credit.
2. Settings → SSH → upload `~/.ssh/id_ed25519.pub`.
3. Settings → Storage → 50 GB Network Volume named `stl-seed-vol` in
   the same region as your pods (US-CA, US-OR, EU-RO are typically
   cheapest for 4090s). The volume persists across spot interruptions
   so you don't re-download Qwen3 weights every time.

## 3. Create the pod

**Option A (first run):** stock template
`runpod/pytorch:2.5.1-py3.11-cuda12.4.1-ubuntu22.04`, 1× RTX 4090 (24 GB)
spot, 30 GB container disk, mount `stl-seed-vol` at `/workspace`,
3-hour Pod Termination.

**Option B (repeat runs):** pre-built image. Build on any CUDA host
(`docker build -t ghcr.io/aa-alghamdi/stl-seed:phase2 -f docker/Dockerfile .`),
push to GHCR, then RunPod → Custom Container with that image. Saves
~5 min of cold-start setup per pod.

The stock template's base image matches `BASE_IMAGE` in
`docker/Dockerfile`, so option A → B is binary compatible.

## 4. Bootstrap

```bash
ssh root@<pod-ip> -p <pod-port> -i ~/.ssh/id_ed25519
cd /workspace
git clone https://github.com/AA-Alghamdi/stl-seed.git
cd stl-seed
bash docker/runpod_bootstrap.sh
```

The bootstrap pins `uv` 0.11.7, runs `uv sync --frozen --extra cuda
--extra dev`, verifies `nvidia-smi` + `torch.cuda.is_available()`,
exercises a tiny NF4 round-trip, and runs `scripts/smoke_test_bnb.py`
(~5 min, ~$0.03). On success it prints `READY FOR PHASE 2`.

Skip the smoke test on subsequent re-runs:

```bash
SKIP_SMOKE_TEST=1 bash docker/runpod_bootstrap.sh
```

## 5. Smoke-test pass criteria

(a) Training runs without crash.
(b) Loss decreases (mean of last window < mean of first window).
(c) No NaN/Inf in loss history.
(d) ≥ 1 of 5 held-out generations parses as the expected XML block.

(d) failing while (a)-(c) pass is *expected* at the 50-step smoke-test
scale — the bnb smoke uses a stripped LoRA (rank 8, q/v only) on 100
rows. Do not raise the smoke budget to chase (d); raise the canonical
sweep budget instead.

## 6. The canonical sweep

### 6.0 Pre-generate trajectories locally (free)

```bash
# On the M5 Pro, NOT on RunPod:
uv run python scripts/generate_canonical.py 2>&1 | tee scripts/generate_canonical.log
```

Writes `data/canonical/<task>/trajectories-*.parquet` for the two
Phase-2 task families (~16 s combined; saves ~30 min of paid GPU
time). Validates two pre-registered gates per task: NaN-drop rate <
1% and ρ-satisfaction fraction ≥ 30%. Exits non-zero if either fails.

`data/canonical/` is gitignored; the sweep runner picks it up via the
resolution order: `data/canonical/` → `data/pilot/` → regenerate
in-cell.

### 6.1 Run

```bash
cd /workspace/stl-seed
uv run python scripts/run_canonical_sweep.py \
    --config-name sweep_main --max-cost-usd 25 --confirm
```

The driver iterates 18 cells in deterministic order, writes
`provenance.json` + `done.flag` per cell, appends one row per cell to
`runs/canonical/sweep_log.csv`. Existing `done.flag` is skipped (spot
recovery). SIGTERM/SIGINT are trapped and logged as `INTERRUPTED`.
`--max-cost-usd` aborts before the next cell if cumulative spend +
estimate would exceed the cap; per-cell estimate is read from
`configs/model/qwen3_*.yaml::expected_minutes_per_cell`.

Optional HF Hub upload: set `hf_hub.enabled: true` and `hf_hub.repo_org`
in `configs/default.yaml`. Without that, adapters stay local under
`runs/canonical/<cell_id>/adapter/`.

Expected wall-clock: ~5-8 hours on a 4090 spot. Cost: ~$3-5 in the
default budget; $25 is the headroom-included ceiling.

### 6.2 Eval + analysis

```bash
uv run python scripts/run_canonical_eval.py --runs-dir runs/canonical
# then locally on the M5 Pro:
uv run python scripts/canonical_analysis.py \
    --eval-results runs/canonical/eval_results.parquet \
    --output-dir runs/canonical/analysis
```

Outputs land in `runs/canonical/analysis/` (posterior.nc, summary
CSVs, TOST results, three figures); the canonical numbers replace the
pilot/smoke placeholders in `paper/results.md`.

Monitoring: `htop` + `nvidia-smi -l 1` in two tmux panes; tail
`runs/sweep_main/cell-*.log`; push a Discord/Slack heartbeat off
`runpod_bootstrap.sh`'s exit.

## 7. Pull artifacts

```bash
scp -r -P <pod-port> -i ~/.ssh/id_ed25519 \
    root@<pod-ip>:/workspace/stl-seed/runs/sweep_main \
    ./runs/runpod-$(date +%Y%m%d)
```

Full sweep dir is ~50 MB (LoRA adapters are tiny relative to the base
model). HF Hub upload (if enabled) means you don't strictly need scp.

## 8. Spot-interruption recovery

1. Pods → New Pod (same image, same volume mount). The volume
   preserves the HF cache, the cloned repo, and partial
   `runs/sweep_main/cell-*` directories.
2. SSH in, `cd /workspace/stl-seed`,
   `SKIP_SMOKE_TEST=1 bash docker/runpod_bootstrap.sh`.
3. Re-launch the sweep with the same flags. The driver detects
   finished cells via `cell-*/done.flag` and skips them.

If you lose >2 hours to interruptions in a row, switch to on-demand
(~3× spot) for the remaining cells.

## 9. Auto-stop / kill switch

Pod Termination is the primary kill switch. Belt-and-braces:

- **In-pod auto-stop:** check elapsed wall-clock at every cell
  boundary; call `runpodctl stop pod $RUNPOD_POD_ID` on budget hit.
- **Heartbeat:** Discord/Slack webhook every 30 min; if the heartbeat
  stops, the pod has been terminated.

## 10. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `uv sync --frozen` hangs ~5 min | First-time wheel download for `torch`+`bitsandbytes`+`triton` (~3 GB) | Patient first boot; instant on subsequent boots if the volume mounts the uv cache |
| `bitsandbytes` import fails ("CUDA Setup failed") | Driver / CUDA mismatch | Pick a different RunPod template, or pin `BNB_CUDA_VERSION=124` (already done in our Dockerfile) |
| `Qwen/Qwen3-0.6B-Instruct` 404s | Model renamed / made gated | `STL_SEED_SMOKE_MODEL=Qwen/Qwen2.5-0.5B-Instruct` |
| `out of memory` on Qwen3-4B | Spot 4090 is 24 GB; max-seq-length too generous | Lower `max_seq_length` from 8192 to 4096 (`configs/model/qwen3_4b.yaml`) |
| HF Hub push fails with 403 | `huggingface-cli login` not run | `huggingface-cli login --token <hf-token>` before the sweep |
| Pod terminates "for no reason" | Pod Termination expired | Extend timer; consider in-pod auto-stop |

## 11. Where to look afterwards

- `paper/smoke_test_bnb_report.md` — first-run bnb verdict.
- `paper/reproducibility.md` — pinned versions, seed propagation, how
  to reproduce.
- `runs/sweep_main/cell-<id>/provenance.json` — per-cell config, loss
  curve, git SHA.
- `runs/sweep_main/cell-<id>/adapter/` — LoRA weights (also on HF Hub
  if upload was enabled).
