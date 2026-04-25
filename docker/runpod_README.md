# RunPod execution playbook for stl-seed Phase 2

This is the operator's manual for running the stl-seed canonical training
sweep on RunPod. It assumes you (Abdullah) have not yet set up a RunPod
account and want a step-by-step procedure with clear cost guardrails.
Phase 1 (everything before this point) was validated on the M5 Pro;
Phase 2 is the first time we spend money.

> **Status as of 2026-04-24:** files in this directory have been authored
> and locally syntax-checked. The Dockerfile has NOT yet been built and the
> bnb smoke test has NOT yet been run on a real GPU — the user has no
> RunPod credit. Ground-truth runtime numbers will be filled into
> `paper/smoke_test_bnb_report.md` on the first real run.

---

## 0. Cost expectations (set them before signing up)

| Item | Hourly | Hours | Cost |
|---|---|---|---|
| Smoke test on 1×4090 spot (this doc, step 6) | $0.34 | 0.5 | $0.17 |
| Canonical 18-cell sweep (3 sizes × 3 filters × 2 tasks) | $0.34 | ~25 | $8.50 |
| Headroom for re-runs / spot interruptions | — | — | $5 |
| Storage (network volume, 50 GB × 30 d) | $0.07/GB-mo | — | $3.50 |
| **Suggested initial credit** | | | **$25** |

These figures use RunPod's published 4090 spot rate of ~$0.34/hr (2026-04
pricing; verify on the dashboard before you commit). If 4090 spot capacity
is gone, the next-cheapest CUDA-12.4-compatible GPU that fits a Qwen3-4B
4-bit base + LoRA training is the A6000 ($0.49/hr spot), or a single A100
40GB ($1.19/hr) if A6000 is also gone.

**Hard budget rule:** never start a pod without setting the auto-stop
timer (RunPod calls this "Pod Termination" — set it to 3 hours initially,
extend manually if needed).

---

## 1. Sign up for RunPod and load credit

1. Create an account at <https://runpod.io>.
2. Settings → Billing → add $25 in credit (Stripe / crypto). RunPod does
   not charge a per-pod fee on top of GPU-hours, so the full balance is
   spendable.
3. Settings → Settings → SSH → upload your public key (the one in
   `~/.ssh/id_ed25519.pub` on the M5 Pro). This makes step 4 ssh-able
   instead of using the in-browser web terminal.
4. Settings → Storage → create a 50 GB **Network Volume** in the same
   region you plan to deploy in (US-CA, US-OR, EU-RO are typically
   cheapest for 4090s). Name it `stl-seed-vol`. The volume persists
   across pod restarts so you do not re-download Qwen3 weights every
   time spot interrupts you.

---

## 2. Create the pod

Two options. The **simple** path uses RunPod's stock PyTorch image and
sets up the env on first boot. The **production** path builds the
Dockerfile in this directory and pushes it to GHCR / Docker Hub for
faster cold-starts. Use option A first; switch to option B once the
sweep is hot enough that the 5-minute first-boot setup cost matters.

### Option A — stock image + bootstrap (recommended for first run)

1. Pods → New Pod.
2. **Template:** "PyTorch 2.5 + CUDA 12.4" (RunPod's official template;
   exact name as of 2026-04 is `runpod/pytorch:2.5.1-py3.11-cuda12.4.1-ubuntu22.04`).
   This base image matches the `BASE_IMAGE` ARG in
   `docker/Dockerfile`, so any subsequent move to option B is binary
   compatible.
3. **GPU:** 1× RTX 4090 (24 GB), spot.
4. **Container Disk:** 30 GB (default fine).
5. **Volume:** mount `stl-seed-vol` at `/workspace`.
6. **Pod Termination:** 3 hours.
7. Create.

### Option B — pre-built image (for repeat runs)

1. On any CUDA host (your own GPU rig, a CI runner, or one bootstrapped
   pod from option A):
   ```bash
   docker build -t ghcr.io/aa-alghamdi/stl-seed:phase2 -f docker/Dockerfile .
   docker push ghcr.io/aa-alghamdi/stl-seed:phase2
   ```
2. RunPod → Pods → New Pod → "Custom Container".
3. Image: `ghcr.io/aa-alghamdi/stl-seed:phase2`.
4. GPU / Volume / Termination: as in option A.
5. Create.

---

## 3. SSH in and clone the repo

```bash
# Find the SSH command on the pod's "Connect" tab in the dashboard.
ssh root@<pod-ip> -p <pod-port> -i ~/.ssh/id_ed25519

# (Option A only — option B already has the repo at /workspace/stl-seed)
cd /workspace
git clone https://github.com/AA-Alghamdi/stl-seed.git
cd stl-seed
git checkout main  # or whatever branch holds the Phase-2 freeze
```

---

## 4. Run the bootstrap (verifies environment)

```bash
cd /workspace/stl-seed
bash docker/runpod_bootstrap.sh
```

This script:

1. Verifies the repo is on a known commit.
2. Checks `uv` is at the pinned version (0.11.7).
3. Runs `uv sync --frozen --extra cuda --extra dev` (no-op on option B
   since the image is pre-synced; ~3 minutes on option A).
4. Confirms `nvidia-smi` exposes a 4090.
5. Confirms `torch.cuda.is_available()` and prints the device name.
6. Loads bitsandbytes and runs a tiny NF4 quantize / dequantize round-trip.
7. Runs `scripts/smoke_test_bnb.py` (~5 minutes, ~$0.03).

If everything passes, you see:

```
================================================================
  READY FOR PHASE 2
================================================================
```

If any step fails, the script exits non-zero with a one-line description
of what broke. Re-read it before re-running — the same env error will
recur.

To skip the smoke test on subsequent re-runs (e.g. after a spot
interruption restart), set `SKIP_SMOKE_TEST=1`:

```bash
SKIP_SMOKE_TEST=1 bash docker/runpod_bootstrap.sh
```

---

## 5. Run the smoke test independently (if desired)

```bash
cd /workspace/stl-seed
uv run python scripts/smoke_test_bnb.py 2>&1 | tee scripts/smoke_test_bnb.log
```

Pass criteria (see `paper/smoke_test_bnb_report.md` after the run):

- (a) Training runs without crash.
- (b) Loss decreases (mean of last window < mean of first window).
- (c) No NaN/Inf in loss history.
- (d) ≥ 1 of 5 held-out generations parses as the expected XML block.

Cost: ~$0.03 (5 minutes on a 4090 spot).

If (d) fails but (a)-(c) pass, that is an *expected* mode at the 50-step
smoke-test scale — the bnb smoke test trains on a smaller LoRA (rank 8,
q/v only) with a stripped-down 100-row dataset, so the model may not yet
emit the right block structure. The smoke test is a *training-loop
sanity*, not a convergence test. Do not raise the smoke-test budget to
chase (d); raise it during the canonical sweep instead.

---

## 6. Run the canonical Phase-2 sweep

The driver is `scripts/run_canonical_sweep.py` (composed via Hydra over
`configs/sweep_main.yaml`). The single-command Phase-2 invocation is:

```bash
cd /workspace/stl-seed
uv run python scripts/run_canonical_sweep.py \
    --config-name sweep_main \
    --max-cost-usd 25 \
    --confirm
```

The sweep iterates the 18 cells (3 sizes × 3 filters × 2 task families)
in deterministic order, writes per-cell `provenance.json` plus a
`done.flag`, and appends one row per cell to
`runs/canonical/sweep_log.csv` with timing + cost telemetry. If a cell's
`done.flag` already exists it is skipped (spot-interruption recovery).
SIGTERM/SIGINT are trapped so pre-emption logs `INTERRUPTED` rather
than crashing.

Cost guardrails:
- `--max-cost-usd 25` aborts before the next cell if cumulative spend
  plus that cell's estimate would exceed the cap.
- Per-cell estimate is read from `configs/model/qwen3_*.yaml`
  (`expected_minutes_per_cell`).

Optional HuggingFace Hub upload: set in `configs/default.yaml` under
`hf_hub:` (`enabled: true`, `repo_org: <your-hf-username>`). Without
those, adapters stay local in `runs/canonical/<cell_id>/adapter/`.

Expected wall-clock: ~5-8 hours on a single 4090 spot (the dry-run
prints a refined cost forecast before any GPU work). Cost: ~$3-5 in the
default budget; $25 is the headroom-included ceiling.

Then run evaluation (~1 hour on the same pod, or locally on the M5 Pro
once adapters are downloaded):

```bash
uv run python scripts/run_canonical_eval.py --runs-dir runs/canonical
```

Then locally back on the M5 Pro (NumPyro fits in CPU minutes for
`scripts/canonical_analysis.py`):

```bash
uv run python scripts/canonical_analysis.py \
    --eval-results runs/canonical/eval_results.parquet \
    --output-dir runs/canonical/analysis
```

Outputs land in `runs/canonical/analysis/` (posterior.nc, summary CSVs,
TOST results, three figures) and the canonical numbers replace the
pilot/smoke placeholders in `paper/results.md` (regenerated on every
run).

Recommended monitoring:

- Open `htop` and `nvidia-smi -l 1` in two tmux panes.
- Tail the per-cell logs: `tail -f runs/sweep_main/cell-*.log`.
- Push a dashboard or Discord/Slack webhook off `runpod_bootstrap.sh`'s
  exit so you get notified of pod termination.

---

## 7. Pull artifacts back

The HF Hub upload in step 6 means you do not need to scp anything —
adapters are public on HF. For the sweep's run directories (loss
curves, per-cell `provenance.json`, `smoke_test_manifest.json`):

```bash
# From the local M5 Pro:
scp -r -P <pod-port> -i ~/.ssh/id_ed25519 \
    root@<pod-ip>:/workspace/stl-seed/runs/sweep_main \
    ./runs/runpod-$(date +%Y%m%d)
```

The full sweep directory is ~50 MB (LoRA adapters are tiny relative to
the base model).

---

## 8. Spot-interruption recovery

Spot pods get pre-empted with no warning. If your pod disappears mid-sweep:

1. Pods → New Pod (same image, same volume mount). The volume persists
   so the HF cache, the cloned repo, and any partially-finished
   `runs/sweep_main/cell-*` directories survive.
2. SSH in, `cd /workspace/stl-seed`.
3. `SKIP_SMOKE_TEST=1 bash docker/runpod_bootstrap.sh` (re-verify env;
   the smoke test is no longer needed since the env is identical to
   the previous boot).
4. Re-launch the sweep with the same flags. The driver detects which
   cells finished (via `cell-*/done.flag` or HF Hub presence) and
   skips them.

If you lose more than 2 hours of work to interruptions in a row, switch
to on-demand pricing for the remaining cells (~3× spot). That's a
budget call: 25 hours of work is $8.50 spot vs ~$25 on-demand.

---

## 9. Auto-stop / kill switch

RunPod's "Pod Termination" timer is the primary kill switch. Belt and
braces on top of it:

1. **In-pod auto-stop:** add a watchdog at the top of your sweep
   driver. Pseudo-code:
   ```python
   import time, subprocess
   _START = time.time()
   _BUDGET_HOURS = 6.0
   def _maybe_die():
       if time.time() - _START > _BUDGET_HOURS * 3600:
           subprocess.run(["runpodctl", "stop", "pod", os.environ["RUNPOD_POD_ID"]])
           raise SystemExit("budget hit; pod terminating")
   ```
   Call `_maybe_die()` at every cell boundary.
2. **Discord/Slack heartbeat:** at the top of the sweep, post a webhook
   message every 30 minutes. If the heartbeat stops, the pod has been
   terminated (either by you or by spot). Lets you intervene before the
   next budget check.

---

## 10. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `uv sync --frozen` hangs ~5 min | First-time wheel download for `torch`+`bitsandbytes`+`triton` (~3 GB) | Be patient; this is one-time per pod boot if no network volume; instant on subsequent boots if the volume mounts the uv cache |
| `bitsandbytes` import fails with "CUDA Setup failed" | Driver / CUDA mismatch (e.g. 4090 host running CUDA 11 driver) | Pick a different RunPod template, or pin `BNB_CUDA_VERSION=124` (already done in our Dockerfile) |
| `Qwen/Qwen3-0.6B-Instruct` 404s | Model has been renamed / made gated since 2026-04 | Set `STL_SEED_SMOKE_MODEL=Qwen/Qwen2.5-0.5B-Instruct` and re-run |
| `out of memory` on Qwen3-4B (largest cell) | Spot 4090 has only 24 GB; max-seq-length too generous | Lower `max_seq_length` from 8192 to 4096 in the Phase-2 config; document the change in `paper/REDACTED.md` |
| HF Hub push fails with 403 | `huggingface-cli login` was not run | `huggingface-cli login --token <hf-token>` at the top of the pod boot, before the sweep |
| Pod terminates "for no reason" | Pod Termination timer expired | Extend the timer; consider the in-pod auto-stop above |

---

## 11. Where to look afterwards

- `paper/smoke_test_bnb_report.md` — first-run bnb verdict (this is what
  you check after step 4 / 5).
- `paper/reproducibility.md` — exact pinned versions, seed propagation,
  how to verify your reproduction.
- `runs/sweep_main/cell-<size>-<filter>-<task>/provenance.json` — per-cell
  config + loss curve + git SHA.
- `runs/sweep_main/cell-<size>-<filter>-<task>/adapter/` — LoRA weights
  (also pushed to HF Hub).

---

## 12. Phase 2 single-command run (TL;DR)

Once `bash docker/runpod_bootstrap.sh` reports `READY FOR PHASE 2`, the
entire Phase-2 path is three commands:

```bash
# On the RunPod pod (after bootstrap.sh passes):
python scripts/run_canonical_sweep.py --config-name sweep_main --max-cost-usd 25 --confirm
# ~5-8 hours wall-clock on a 4090 spot

python scripts/run_canonical_eval.py --runs-dir runs/canonical
# ~1 hour

# Then locally back on the M5 Pro:
python scripts/canonical_analysis.py --eval-results runs/canonical/eval_results.parquet
```

The first command is gated by `--confirm`; without that flag the sweep
prints the cell enumeration and cost forecast then exits. A `--dry-run`
flag is also available for iterating on the configs locally without
spending money.
