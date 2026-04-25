#!/usr/bin/env bash
# stl-seed RunPod bootstrap.
#
# Run this once on a fresh pod to verify the environment is ready for the
# Phase 2 canonical sweep. It is idempotent — safe to re-run after a spot
# interruption.
#
# Verifications, in order:
#   1. Repository present and on a known commit.
#   2. uv on PATH at the pinned version.
#   3. `uv sync --frozen --extra cuda --extra dev` completes (no-op if
#      already synced; brings the env up to date if the user updated
#      uv.lock between pod runs).
#   4. nvidia-smi exposes at least one CUDA device.
#   5. torch.cuda.is_available() is True.
#   6. bitsandbytes loads its NF4 kernel (fail-fast if the driver/CUDA
#      mismatch breaks bnb at import time).
#   7. Run the bnb 1-min smoke test (scripts/smoke_test_bnb.py). Pass
#      criteria mirror the MLX smoke test (paper/REDACTED.md):
#      loss decreases, no NaN, parse-success >= 1/5.
#
# On all-pass, prints a green "READY FOR PHASE 2" banner and exits 0.
# On any failure, prints which step failed and exits non-zero.
#
# Usage:
#   bash docker/runpod_bootstrap.sh
#   # or, with a different repo root:
#   STL_SEED_ROOT=/workspace/stl-seed bash docker/runpod_bootstrap.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------

STL_SEED_ROOT="${STL_SEED_ROOT:-/workspace/stl-seed}"
EXPECTED_UV_VERSION="${EXPECTED_UV_VERSION:-0.11.7}"
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"

# ANSI colors (skip if not a tty so logs stay clean).
if [[ -t 1 ]]; then
    GREEN=$'\033[0;32m'
    RED=$'\033[0;31m'
    YELLOW=$'\033[0;33m'
    BOLD=$'\033[1m'
    RESET=$'\033[0m'
else
    GREEN=""
    RED=""
    YELLOW=""
    BOLD=""
    RESET=""
fi

step() {
    printf "%s==>%s %s%s%s\n" "$YELLOW" "$RESET" "$BOLD" "$1" "$RESET"
}

ok() {
    printf "  %sOK%s   %s\n" "$GREEN" "$RESET" "$1"
}

fail() {
    printf "  %sFAIL%s %s\n" "$RED" "$RESET" "$1" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# 1. Repo present.
# ---------------------------------------------------------------------------

step "1/7 Repository sanity"
if [[ ! -d "$STL_SEED_ROOT" ]]; then
    fail "STL_SEED_ROOT=$STL_SEED_ROOT does not exist. Clone or mount the repo first."
fi
cd "$STL_SEED_ROOT"
if [[ ! -f pyproject.toml ]]; then
    fail "pyproject.toml not found in $STL_SEED_ROOT — wrong directory?"
fi
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
HEAD_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo detached)"
ok "repo at $STL_SEED_ROOT (branch=$HEAD_BRANCH, sha=${HEAD_SHA:0:8})"

# ---------------------------------------------------------------------------
# 2. uv pinned version.
# ---------------------------------------------------------------------------

step "2/7 uv toolchain"
if ! command -v uv >/dev/null 2>&1; then
    fail "uv not on PATH — re-build the docker image or run install.sh manually."
fi
ACTUAL_UV_VERSION="$(uv --version | awk '{print $2}')"
if [[ "$ACTUAL_UV_VERSION" != "$EXPECTED_UV_VERSION" ]]; then
    printf "  %sWARN%s uv version mismatch: expected %s, got %s\n" \
        "$YELLOW" "$RESET" "$EXPECTED_UV_VERSION" "$ACTUAL_UV_VERSION"
fi
ok "uv $ACTUAL_UV_VERSION"

# ---------------------------------------------------------------------------
# 3. uv sync.
# ---------------------------------------------------------------------------

step "3/7 Python deps (uv sync --frozen)"
# --frozen ensures we use exactly the pinned versions in uv.lock; no
# silent upgrades. The cuda extra installs torch + bitsandbytes + trl;
# dev gives us pytest for the in-pod test suite.
uv sync --frozen --extra cuda --extra dev >/tmp/uv-sync.log 2>&1 \
    || { tail -40 /tmp/uv-sync.log >&2; fail "uv sync failed — see /tmp/uv-sync.log"; }
ok "uv sync clean (log: /tmp/uv-sync.log)"

# ---------------------------------------------------------------------------
# 4. nvidia-smi.
# ---------------------------------------------------------------------------

step "4/7 nvidia-smi"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found — pod has no GPU runtime mounted."
fi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
ok "GPU=$GPU_NAME mem=$GPU_MEM driver=$DRIVER"

# ---------------------------------------------------------------------------
# 5. torch.cuda.is_available.
# ---------------------------------------------------------------------------

step "5/7 PyTorch CUDA"
uv run python - <<'PY' || fail "torch.cuda.is_available() returned False."
import torch
assert torch.cuda.is_available(), "no CUDA device visible to torch"
print(f"  torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
PY
ok "torch sees CUDA"

# ---------------------------------------------------------------------------
# 6. bitsandbytes import + NF4 kernel sanity.
# ---------------------------------------------------------------------------

step "6/7 bitsandbytes NF4"
# Loading bnb on a CUDA-mismatched driver tends to fail at import with
# "CUDA Setup failed despite CUDA being available." We surface that clearly
# instead of letting it die mid-training. We also do a minimal NF4 quantize
# round-trip to confirm the kernel actually executes.
uv run python - <<'PY' || fail "bitsandbytes NF4 kernel did not load — likely CUDA/driver mismatch."
import torch
import bitsandbytes as bnb
from bitsandbytes.functional import quantize_nf4, dequantize_nf4

# Tiny tensor; this exercises the same kernel path the bnb 4-bit Linear
# layer does on the first forward pass.
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
q, qstate = quantize_nf4(x)
xr = dequantize_nf4(q, qstate)
err = (x - xr).abs().mean().item()
print(f"  bnb={bnb.__version__}; NF4 round-trip mean abs error={err:.4e}")
# NF4 is lossy by design; the round-trip error should be well under 0.1.
assert err < 0.1, f"NF4 round-trip error {err} suspiciously large"
PY
ok "bnb NF4 kernel functional"

# ---------------------------------------------------------------------------
# 7. End-to-end bnb smoke test.
# ---------------------------------------------------------------------------

if [[ "$SKIP_SMOKE_TEST" == "1" ]]; then
    printf "  %sSKIP%s smoke test (SKIP_SMOKE_TEST=1)\n" "$YELLOW" "$RESET"
else
    step "7/7 bnb QLoRA smoke test (~5 min, ~$0.10)"
    # The script writes its own report; we surface its exit code only.
    if ! uv run python scripts/smoke_test_bnb.py; then
        fail "scripts/smoke_test_bnb.py did not pass — see paper/smoke_test_bnb_report.md"
    fi
    ok "bnb smoke test passed"
fi

# ---------------------------------------------------------------------------
# Banner.
# ---------------------------------------------------------------------------

cat <<EOF

${GREEN}${BOLD}================================================================${RESET}
${GREEN}${BOLD}  READY FOR PHASE 2${RESET}
${GREEN}${BOLD}================================================================${RESET}
  repo:    $STL_SEED_ROOT @ ${HEAD_SHA:0:8} (${HEAD_BRANCH})
  uv:      $ACTUAL_UV_VERSION
  GPU:     $GPU_NAME ($GPU_MEM, driver $DRIVER)

  Next: launch the canonical sweep with the Phase 2 driver
  (to be added under scripts/ before kicking off the real run).

  Reminder: this is a SPOT pod — set an auto-stop budget and
  push checkpoints to HF Hub frequently.
EOF
