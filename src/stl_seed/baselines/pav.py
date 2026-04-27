"""Process Advantage Verifier (PAV; Setlur et al. 2024, arXiv:2410.08146).

Background
----------
Setlur, Nagpal, Fisch, Geng, Eisenstein, Agarwal, Agarwal, Berant & Kumar
("Rewarding Progress: Scaling Automated Process Verifiers", arXiv:2410.08146)
introduce the *Process Advantage Verifier*: a learned per-step reward
model that predicts the change in expected success probability before vs.
after a step, trained on Monte-Carlo continuation labels. The PAV is the
strongest learned process-reward baseline in their evaluation; it is the
correct point of comparison for a verifier that claims to be a better
process signal than learned PRMs.

In stl-seed, we use PAV as the *learned* counterpart to the formal
STL-rho verifier in :mod:`stl_seed.stl.evaluator`. Both consume a
trajectory; STL-rho returns a closed-form Donzé-Maler robustness;
PAV returns a sum of learned per-step advantages. The empirical question
we ask in :func:`stl_seed.baselines.comparison.compare_pav_vs_stl` is
whether either signal predicts terminal success better on a held-out
split, and how much training data PAV needs to match STL.

Architecture
------------
The MLP is a 3-layer Equinox feed-forward network:

    Linear(state_dim -> hidden) -> ReLU -> Dropout(p)
    Linear(hidden     -> hidden) -> ReLU -> Dropout(p)
    Linear(hidden     -> 1)              # scalar advantage prediction

Default ``hidden=256``, ``dropout=0.1``. With ``model_selection=True`` on
:meth:`PAVProcessRewardModel.fit_with_selection`, we sweep
``hidden in {64, 128, 256, 512}`` x ``weight_decay in {0, 1e-4, 1e-3, 1e-2}``
and pick the (width, wd) pair with the lowest *trajectory-disjoint* val MSE
(Setlur §A.3 reports an analogous sweep). Equinox is consistent with the
rest of the JAX-native stack; we deliberately avoid PyTorch.

Per-step MC labels: two variants
--------------------------------
Setlur et al. label step t with the *advantage*:

    A_t  =  P_succeed(state after step t)  -  P_succeed(state before step t)

P_succeed at a state is the empirical fraction of *continuations* from that
state that succeed at the terminal.

* **kNN variant** (this module's ``compute_per_step_mc_labels``).
  Estimates P_succeed at state s_t via nearest-neighbor pooling against
  the corpus's terminal-success vector (Manhattan distance in z-scored
  state space). Zero additional simulation cost; the approximation is
  documented at length below. Used to be the only option, and the audit
  on 2026-04-26 flagged it as a deliberate weakening relative to Setlur.

* **On-policy rollout variant** (``stl_seed.baselines.pav_rollout``).
  At each prefix length t, freezes the canonical actions for steps
  ``0..t-1``, draws ``K`` random tails for steps ``t..H-1``, re-integrates
  the simulator, and reports ``mean_k 1{rho(traj_k) > 0}``. This is the
  Setlur §3.2 "fresh continuations" estimator, made possible only because
  the canonical store records the seed and policy; we reconstruct the
  simulator deterministically from the metadata. More expensive
  (``O(N_traj * H * K)`` ODE integrations vs zero for the kNN variant)
  but is what the Setlur paper actually reports.

When you see ``pav_v2.md``, the on-policy variant is what is being
compared. The kNN variant is kept for the legacy ``pav_comparison.md``
artifact and for synthetic unit tests where there is no simulator to
re-roll from.

Training
--------
``fit(...)`` minimizes mean-squared error between predicted and target
per-step advantages, with per-trajectory random partitions used to mint
the (state, advantage) training pairs. The loss history is returned so the
caller can validate convergence. Adam optimizer (lr=1e-3 default), full-
batch by default for stability on the small (typically O(10k)) (state,
advantage) sets we work with. ``fit_with_selection(...)`` extends this with
early-stopping on val MSE plateau (patience configurable) and a
hidden-width / weight-decay grid search.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from stl_seed.tasks._trajectory import Trajectory

# ---------------------------------------------------------------------------
# Per-step Monte-Carlo continuation labels.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepDataset:
    """Flat (state, advantage) dataset extracted from a trajectory corpus.

    Fields
    ------
    states:
        Shape ``(N_steps, state_dim)``. Per-step state vectors at the
        beginning of each step, stacked across the input trajectories.
    advantages:
        Shape ``(N_steps,)``. Per-step advantage A_t = MC_after - MC_before.
    traj_ids:
        Shape ``(N_steps,)``. Index of the trajectory each step came from
        (lets the trainer take per-trajectory train/val splits).
    step_ids:
        Shape ``(N_steps,)``. Index of the step within its trajectory.
    """

    states: np.ndarray  # shape (N_steps, state_dim), float32
    advantages: np.ndarray  # shape (N_steps,), float32
    traj_ids: np.ndarray  # shape (N_steps,), int64
    step_ids: np.ndarray  # shape (N_steps,), int64


def _state_at_step(traj: Trajectory, k: int, n_steps: int) -> np.ndarray:
    """Return the state vector at the beginning of action-step k.

    The trajectory has ``T`` save-points and ``H`` action-steps. We map
    action-step k (in [0, n_steps]) to a save-point index proportionally,
    so step 0 is the initial state and step n_steps is the terminal state.
    """
    T = int(np.asarray(traj.states).shape[0])
    if T == 0:
        raise ValueError("Trajectory has zero save-points.")
    # Index 0 -> 0; index n_steps -> T-1; linearly interpolate in between.
    idx = 0 if n_steps <= 0 else int(round(k * (T - 1) / max(n_steps, 1)))
    idx = max(0, min(idx, T - 1))
    return np.asarray(traj.states)[idx].astype(np.float64)


def _knn_success_probability(
    query_states: np.ndarray,
    pool_states: np.ndarray,
    pool_success: np.ndarray,
    k: int,
) -> np.ndarray:
    """Estimate empirical P(success | state) via k-nearest-neighbor lookup.

    For each row in ``query_states``, find the k closest rows in
    ``pool_states`` (Manhattan distance in z-scored space), then return the
    mean of the corresponding ``pool_success`` entries.

    Manhattan + z-scoring is robust to disparate scales across state
    channels (concentrations, fractions, glucose mg/dL all on different
    scales in this codebase). k defaults are set in ``compute_per_step_mc_labels``.

    Implementation note: O(Q * P * D) is fine for the ~5k-step pools we
    work with on M-series; we use NumPy broadcasting for clarity.
    """
    if pool_states.shape[0] == 0:
        return np.full(query_states.shape[0], 0.5, dtype=np.float64)
    # Z-score against the *pool* so query and pool live in comparable units.
    mu = pool_states.mean(axis=0, keepdims=True)
    sd = pool_states.std(axis=0, keepdims=True) + 1e-9
    pool_z = (pool_states - mu) / sd
    query_z = (query_states - mu) / sd
    out = np.empty(query_states.shape[0], dtype=np.float64)
    eff_k = min(k, pool_states.shape[0])
    for i, q in enumerate(query_z):
        d = np.abs(pool_z - q[None, :]).sum(axis=1)
        nn_idx = np.argpartition(d, eff_k - 1)[:eff_k]
        out[i] = float(pool_success[nn_idx].mean())
    return out


def compute_per_step_mc_labels(
    trajectories: list[Trajectory],
    terminal_success: jnp.ndarray | np.ndarray,
    k_neighbors: int = 16,
    n_steps: int | None = None,
) -> StepDataset:
    """Build per-step (state, MC-advantage) training pairs from a corpus.

    Parameters
    ----------
    trajectories:
        List of :class:`Trajectory` objects (the canonical corpus).
    terminal_success:
        Shape ``(N_traj,)`` 0/1 array; ``terminal_success[i] == 1`` iff
        ``trajectories[i]`` satisfies the spec at the terminal.
    k_neighbors:
        Number of nearest neighbors to pool for the empirical
        success-probability estimate (Setlur §3.2 uses M ≈ 8-32 fresh
        rollouts per state; our k_neighbors is the offline analogue).
    n_steps:
        Number of action-steps per trajectory. If ``None`` we use the
        trajectory's ``actions.shape[0]`` (the canonical horizon).

    Returns
    -------
    StepDataset:
        Flat (state, advantage) pairs. ``states[i]`` is the pre-step
        state vector and ``advantages[i] = MC(s_{t+1}) - MC(s_t)`` is
        the empirical advantage of the step that produced ``s_{t+1}``.

    Notes
    -----
    The MC estimator P_succeed(s) is the kNN average of terminal-success
    indicators among the k nearest *step-matched* states in the pool.
    "Step-matched" means: when estimating P_succeed at action-step t in
    trajectory i, we restrict the pool to (state_at_step_t, terminal_succ)
    pairs across *all other* trajectories. This isolates the temporal
    contribution of the step itself and is the offline analogue of Setlur
    et al.'s on-policy rollout protocol.

    For the terminal state (k = n_steps) we set MC = terminal_success
    directly (the indicator is the ground truth).
    """
    ts = np.asarray(terminal_success, dtype=np.float64)
    if ts.shape[0] != len(trajectories):
        raise ValueError(
            f"terminal_success length {ts.shape[0]} != n_trajectories {len(trajectories)}"
        )
    if len(trajectories) == 0:
        raise ValueError("Need at least one trajectory.")

    # Determine horizon.
    horizons = {int(np.asarray(t.actions).shape[0]) for t in trajectories}
    if len(horizons) > 1:
        raise ValueError(f"All trajectories must share a horizon; got {sorted(horizons)}")
    H = next(iter(horizons)) if n_steps is None else int(n_steps)

    state_dim = int(np.asarray(trajectories[0].states).shape[1])

    # Stack per-step states for every (trajectory, step) into
    # one big tensor of shape (N_traj, H+1, state_dim) so we can
    # build per-step pools cheaply.
    N = len(trajectories)
    step_states = np.zeros((N, H + 1, state_dim), dtype=np.float64)
    for i, traj in enumerate(trajectories):
        for k in range(H + 1):
            step_states[i, k] = _state_at_step(traj, k, H)

    # Per-step MC estimates: shape (N, H+1).
    mc = np.zeros((N, H + 1), dtype=np.float64)
    # Terminal step is exact.
    mc[:, H] = ts
    # Pre-terminal steps: per-step kNN pool using leave-one-out on the
    # query's own trajectory.
    for k in range(H):
        pool_states_all = step_states[:, k, :]  # (N, D)
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            pool = pool_states_all[mask]
            pool_y = ts[mask]
            mc[i, k] = float(
                _knn_success_probability(
                    query_states=step_states[i, k][None, :],
                    pool_states=pool,
                    pool_success=pool_y,
                    k=k_neighbors,
                )[0]
            )

    # Advantages: A_{i, k} = mc[i, k+1] - mc[i, k], k = 0 .. H-1.
    # The state we *condition on* at step k is the pre-step state,
    # i.e. step_states[i, k]. This matches Setlur §3.2.
    flat_states = step_states[:, :H, :].reshape(-1, state_dim)
    flat_adv = (mc[:, 1:] - mc[:, :H]).reshape(-1)
    flat_traj = np.repeat(np.arange(N, dtype=np.int64), H)
    flat_step = np.tile(np.arange(H, dtype=np.int64), N)

    return StepDataset(
        states=flat_states.astype(np.float32),
        advantages=flat_adv.astype(np.float32),
        traj_ids=flat_traj,
        step_ids=flat_step,
    )


# ---------------------------------------------------------------------------
# Equinox MLP.
# ---------------------------------------------------------------------------


class _MLP(eqx.Module):
    """3-layer feed-forward network with ReLU + dropout, scalar output.

    Uses Equinox to stay in the JAX pytree world (matches the rest of the
    codebase; no PyTorch). Dropout is applied only during training (the
    ``training`` flag in :meth:`__call__` controls it).
    """

    layers: tuple[eqx.nn.Linear, eqx.nn.Linear, eqx.nn.Linear]
    dropout: float = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        hidden: int,
        dropout: float,
        *,
        key: jax.Array,
    ) -> None:
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = (
            eqx.nn.Linear(state_dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
            eqx.nn.Linear(hidden, 1, key=k3),
        )
        self.dropout = float(dropout)

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        training: bool = False,
    ) -> jax.Array:
        h = jax.nn.relu(self.layers[0](x))
        if training and self.dropout > 0.0:
            assert key is not None, "training=True requires a PRNG key"
            k1, k2 = jax.random.split(key)
            h = _apply_dropout(h, self.dropout, k1)
        h = jax.nn.relu(self.layers[1](h))
        if training and self.dropout > 0.0:
            assert key is not None
            h = _apply_dropout(h, self.dropout, k2)
        return self.layers[2](h)[0]


def _apply_dropout(x: jax.Array, p: float, key: jax.Array) -> jax.Array:
    keep = 1.0 - p
    mask = jax.random.bernoulli(key, p=keep, shape=x.shape).astype(x.dtype)
    return (x * mask) / keep


# ---------------------------------------------------------------------------
# PAV public API.
# ---------------------------------------------------------------------------


@dataclass
class PAVTrainingHistory:
    """Loss curves and timing for a single ``fit`` call."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    wall_time_s: float = 0.0
    n_train_pairs: int = 0
    n_val_pairs: int = 0


class PAVProcessRewardModel:
    """Process Advantage Verifier (Setlur et al. 2024, arXiv:2410.08146).

    A small Equinox MLP trained to predict per-step "advantage" ---
    the change in expected success probability before vs. after a step
    --- from per-step Monte-Carlo continuation labels.

    Training pipeline
    -----------------
    1. From a corpus of trajectories with terminal success labels:
       * Compute per-step MC success probability via empirical kNN
         continuation success on the corpus (see
         :func:`compute_per_step_mc_labels`).
       * Compute advantage = MC_after - MC_before per step.
    2. Train MLP(state -> advantage) on the (state, advantage) pairs.
    3. Use the trained MLP to score new trajectories per-step and sum
       the predicted advantages.

    Architecture
    ------------
    3-layer MLP, hidden 256, ReLU, dropout 0.1 (defaults).

    Parameters
    ----------
    state_dim:
        Dimension of the per-step state vector. Must match
        ``trajectory.states.shape[1]`` for any trajectory passed to
        :meth:`score`.
    hidden:
        Hidden width of each fully-connected layer.
    dropout:
        Dropout probability applied between layers during training.
    """

    def __init__(self, state_dim: int, hidden: int = 256, dropout: float = 0.1) -> None:
        self.state_dim = int(state_dim)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self._model: _MLP | None = None
        self._state_mu: np.ndarray | None = None
        self._state_sd: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        trajectories: list[Trajectory],
        terminal_success: jnp.ndarray | np.ndarray,
        n_epochs: int = 50,
        lr: float = 1e-3,
        key: jax.Array | None = None,
        batch_size: int | None = None,
        val_frac: float = 0.1,
        k_neighbors: int = 16,
        verbose: bool = False,
        weight_decay: float = 0.0,
        early_stopping_patience: int | None = None,
        precomputed_train: StepDataset | None = None,
        precomputed_val: StepDataset | None = None,
    ) -> dict[str, list[float] | float | int]:
        """Train the PAV. Returns a dict with loss history and timing.

        Parameters
        ----------
        trajectories:
            Training corpus.
        terminal_success:
            ``(N_traj,)`` 0/1 indicators (1 iff terminal state satisfies the
            target spec).
        n_epochs:
            Number of full-batch passes over the (state, advantage) set.
        lr:
            Adam learning rate.
        key:
            JAX PRNG key. Defaults to ``jax.random.PRNGKey(0)``.
        batch_size:
            Mini-batch size. ``None`` -> full batch (deterministic, robust on
            the small datasets we typically have).
        val_frac:
            Fraction of *trajectories* (not steps) held out for validation
            loss reporting. The split is done at the trajectory level so the
            kNN MC estimator does not leak step-level information across
            split.
        k_neighbors:
            Pool size for the kNN MC estimator (see
            :func:`compute_per_step_mc_labels`).
        verbose:
            Print per-epoch losses to stdout.
        weight_decay:
            Decoupled-weight-decay coefficient (AdamW-style; default 0.0
            preserves the previous Adam behavior). Applied to *all* array
            leaves of the model pytree at each Adam step.
        early_stopping_patience:
            If set and a validation set exists, stop after this many
            consecutive epochs with no improvement on val MSE *and* restore
            the best-val-MSE model checkpoint. ``None`` -> disabled
            (run all ``n_epochs``, retain the final-epoch model).
        precomputed_train, precomputed_val:
            Optional pre-built :class:`StepDataset` objects. When supplied,
            we skip the kNN MC label computation entirely and use the
            given (state, advantage) pairs. This lets a caller plug in
            on-policy-rollout MC labels (see
            :mod:`stl_seed.baselines.pav_rollout`) or amortize a single
            label computation across a hyperparameter sweep. When
            ``precomputed_train`` is given, ``trajectories`` and
            ``terminal_success`` are still required for shape-validation
            and the trajectory-level train/val split bookkeeping but the
            MC labels they would otherwise produce are discarded.

        Returns
        -------
        dict with keys:
            ``train_loss``: list[float], one per epoch (MSE).
            ``val_loss``: list[float], one per epoch.
            ``wall_time_s``: float, seconds for the entire fit call.
            ``n_train_pairs``: int.
            ``n_val_pairs``: int.
            ``best_epoch``: int, the (1-indexed) epoch with the lowest val
                MSE, or ``n_epochs`` if no val set / no improvement.
            ``stopped_early``: bool, whether training terminated before
                ``n_epochs`` due to early-stopping patience exhaustion.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        ts = np.asarray(terminal_success, dtype=np.float64)
        n_traj = len(trajectories)
        if n_traj < 2:
            raise ValueError(f"PAV.fit needs >=2 trajectories to form a kNN pool; got {n_traj}")

        if precomputed_train is not None:
            # Caller has already minted (state, advantage) pairs. The
            # trajectory-level split is the caller's responsibility; we
            # take what we are given.
            train_ds = precomputed_train
            val_ds = precomputed_val
        else:
            # Trajectory-level train/val split (preserves leave-one-out
            # semantics of the kNN estimator within each side).
            rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
            perm = rng.permutation(n_traj)
            n_val = max(1, int(round(val_frac * n_traj))) if val_frac > 0.0 else 0
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]
            if len(tr_idx) < 2:
                # Not enough train trajectories. collapse split so kNN works.
                tr_idx = perm
                val_idx = np.array([], dtype=np.int64)

            train_trajs = [trajectories[int(i)] for i in tr_idx]
            train_ts = ts[tr_idx]
            val_trajs = [trajectories[int(i)] for i in val_idx]
            val_ts = ts[val_idx]

            train_ds = compute_per_step_mc_labels(train_trajs, train_ts, k_neighbors=k_neighbors)
            val_ds = None
            if len(val_trajs) >= 2:
                val_ds = compute_per_step_mc_labels(
                    val_trajs, val_ts, k_neighbors=min(k_neighbors, len(val_trajs) - 1)
                )

        # Standardize states to roughly unit scale to stabilize SGD across
        # task families with very different state scales.
        mu = train_ds.states.mean(axis=0)
        sd = train_ds.states.std(axis=0) + 1e-6
        self._state_mu = mu.astype(np.float32)
        self._state_sd = sd.astype(np.float32)

        train_x = (train_ds.states - mu) / sd
        train_y = train_ds.advantages
        val_x = None
        val_y = None
        if val_ds is not None:
            val_x = (val_ds.states - mu) / sd
            val_y = val_ds.advantages

        key, init_key = jax.random.split(key)
        model = _MLP(
            state_dim=self.state_dim,
            hidden=self.hidden,
            dropout=self.dropout,
            key=init_key,
        )

        # Hand-rolled AdamW to avoid taking on optax as a dependency
        # (it's a separate package and not in pyproject.toml). When
        # weight_decay == 0 this collapses to plain Adam.
        opt_state = _adam_init(model)

        if batch_size is None or batch_size >= train_x.shape[0]:
            batch_size = train_x.shape[0]

        history = PAVTrainingHistory(
            n_train_pairs=int(train_x.shape[0]),
            n_val_pairs=int(val_x.shape[0]) if val_x is not None else 0,
        )

        t0 = time.time()
        n_train = train_x.shape[0]

        train_x_j = jnp.asarray(train_x)
        train_y_j = jnp.asarray(train_y)
        val_x_j = jnp.asarray(val_x) if val_x is not None else None
        val_y_j = jnp.asarray(val_y) if val_y is not None else None

        # Best-val-MSE tracking: keep the model at the lowest val_loss and
        # restore it at the end if early stopping triggered (or even if it
        # didn't. restoring the best is always at-least-as-good in
        # generalization terms when there is val signal).
        best_val = float("inf")
        best_model = model
        best_epoch = int(n_epochs)
        bad_epochs = 0
        stopped_early = False

        for epoch in range(int(n_epochs)):
            key, perm_key, drop_key = jax.random.split(key, 3)
            perm_idx = jax.random.permutation(perm_key, n_train)
            xs = train_x_j[perm_idx]
            ys = train_y_j[perm_idx]
            epoch_losses = []
            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                xb = xs[start:end]
                yb = ys[start:end]
                drop_key, sub = jax.random.split(drop_key)
                loss, grads = _loss_and_grad(model, xb, yb, sub, training=True)
                model, opt_state = _adam_step(
                    model, grads, opt_state, lr=float(lr), weight_decay=float(weight_decay)
                )
                epoch_losses.append(float(loss))
            train_loss = float(np.mean(epoch_losses))
            history.train_loss.append(train_loss)

            if val_x_j is not None:
                v_loss = float(_eval_loss(model, val_x_j, val_y_j))
                history.val_loss.append(v_loss)
                # Track best.
                if v_loss < best_val - 1e-9:
                    best_val = v_loss
                    best_model = model
                    best_epoch = epoch + 1
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                # Early stopping check.
                if early_stopping_patience is not None and bad_epochs >= int(
                    early_stopping_patience
                ):
                    if verbose:
                        print(
                            f"[PAV] early-stop at epoch {epoch + 1}/{n_epochs} "
                            f"(best val_mse={best_val:.6f} @ epoch {best_epoch})"
                        )
                    stopped_early = True
                    break
            else:
                history.val_loss.append(float("nan"))
                # No val set: keep the most recent model as "best".
                best_model = model
                best_epoch = epoch + 1

            if verbose:
                print(
                    f"[PAV] epoch {epoch + 1}/{n_epochs} "
                    f"train_mse={train_loss:.6f} val_mse={history.val_loss[-1]:.6f}"
                )

        history.wall_time_s = float(time.time() - t0)
        # Restore the best-val checkpoint (a no-op if val was disabled).
        self._model = best_model
        return {
            "train_loss": list(history.train_loss),
            "val_loss": list(history.val_loss),
            "wall_time_s": float(history.wall_time_s),
            "n_train_pairs": int(history.n_train_pairs),
            "n_val_pairs": int(history.n_val_pairs),
            "best_epoch": int(best_epoch),
            "stopped_early": bool(stopped_early),
        }

    # ----------------------------------------------------- fit_with_selection
    @classmethod
    def fit_with_selection(
        cls,
        trajectories: list[Trajectory],
        terminal_success: jnp.ndarray | np.ndarray,
        state_dim: int,
        *,
        hidden_grid: tuple[int, ...] = (64, 128, 256, 512),
        weight_decay_grid: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2),
        dropout: float = 0.1,
        n_epochs: int = 100,
        lr: float = 1e-3,
        key: jax.Array | None = None,
        val_frac: float = 0.2,
        k_neighbors: int = 16,
        early_stopping_patience: int = 5,
        precomputed_train: StepDataset | None = None,
        precomputed_val: StepDataset | None = None,
        verbose: bool = False,
    ) -> tuple[PAVProcessRewardModel, dict[str, Any]]:
        """Hidden-width / weight-decay grid search with held-out val MSE.

        For each ``(hidden, weight_decay)`` cell we fresh-init an MLP, fit
        with early-stopping-on-val-MSE, and record the best-epoch val MSE.
        We pick the ``(hidden, weight_decay)`` with the lowest val MSE,
        re-instantiate the PAV at that width, and refit it (with the same
        early-stopping criterion) so the returned model is the one
        callers will :meth:`score` with.

        The MC labels are computed *once* (or supplied via the
        ``precomputed_*`` arguments) and reused across every cell of the
        grid; only the MLP fit is repeated. This makes the sweep cost
        roughly ``|hidden_grid| * |weight_decay_grid|`` epochs, not
        kNN-label-computation passes.

        Parameters
        ----------
        trajectories, terminal_success, state_dim:
            Same meaning as :meth:`fit`. ``state_dim`` is required because
            we build a fresh MLP per grid cell.
        hidden_grid, weight_decay_grid:
            Hyperparameter grids. Defaults match the audit
            (2026-04-26): ``hidden in {64, 128, 256, 512}``,
            ``wd in {0, 1e-4, 1e-3, 1e-2}``.
        dropout:
            Held fixed at ``dropout=0.1`` per Setlur §A.3.
        n_epochs, lr, val_frac, k_neighbors, early_stopping_patience:
            Forwarded to each :meth:`fit` call.
        precomputed_train, precomputed_val:
            Optional one-time-built (state, advantage) datasets. When
            supplied we skip the kNN label computation in *every* fit
            call. This is the main lever that makes the grid search
            affordable on the canonical corpus.
        verbose:
            Print per-cell val MSE.

        Returns
        -------
        (best_pav, selection_report) where ``best_pav`` is a fitted
        :class:`PAVProcessRewardModel` at the best ``(hidden, wd)`` and
        ``selection_report`` is a dict with keys:
            ``best_hidden``: int.
            ``best_weight_decay``: float.
            ``best_val_mse``: float.
            ``best_epoch``: int (1-indexed) at the chosen cell.
            ``grid``: list of dicts, one per cell, with
                (hidden, weight_decay, best_val_mse, best_epoch,
                 stopped_early, n_epochs_run, wall_time_s).
            ``total_wall_time_s``: float.
            ``final_history``: the loss history of the final refit
                (for downstream plotting).
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        if not hidden_grid:
            raise ValueError("hidden_grid must be non-empty")
        if not weight_decay_grid:
            raise ValueError("weight_decay_grid must be non-empty")

        # If labels are not supplied, compute them once with the
        # trajectory-level split. We then pass these through to every
        # subsequent fit() so the MC labels and split are *identical*
        # across the grid (apples-to-apples model selection).
        if precomputed_train is None:
            ts_arr = np.asarray(terminal_success, dtype=np.float64)
            n_traj = len(trajectories)
            if n_traj < 2:
                raise ValueError(f"fit_with_selection needs >=2 trajectories; got {n_traj}")
            split_key, _ = jax.random.split(key)
            split_seed = int(jax.random.randint(split_key, (), 0, 2**31 - 1))
            rng = np.random.default_rng(split_seed)
            perm = rng.permutation(n_traj)
            n_val = max(2, int(round(val_frac * n_traj))) if val_frac > 0.0 else 0
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]
            if len(tr_idx) < 2:
                tr_idx = perm
                val_idx = np.array([], dtype=np.int64)

            train_trajs = [trajectories[int(i)] for i in tr_idx]
            train_ts = ts_arr[tr_idx]
            val_trajs = [trajectories[int(i)] for i in val_idx]
            val_ts = ts_arr[val_idx]

            precomputed_train = compute_per_step_mc_labels(
                train_trajs, train_ts, k_neighbors=k_neighbors
            )
            if len(val_trajs) >= 2:
                precomputed_val = compute_per_step_mc_labels(
                    val_trajs, val_ts, k_neighbors=min(k_neighbors, len(val_trajs) - 1)
                )

        if precomputed_val is None:
            raise ValueError(
                "fit_with_selection requires a non-empty val set "
                "(model selection without val MSE is not meaningful)"
            )

        # Sweep.
        t_total = time.time()
        grid_records: list[dict[str, Any]] = []
        best_cell: dict[str, Any] | None = None
        best_val = float("inf")
        for hidden in hidden_grid:
            for wd in weight_decay_grid:
                key, sub_key = jax.random.split(key)
                pav = cls(state_dim=int(state_dim), hidden=int(hidden), dropout=float(dropout))
                hist = pav.fit(
                    trajectories=trajectories,
                    terminal_success=terminal_success,
                    n_epochs=int(n_epochs),
                    lr=float(lr),
                    key=sub_key,
                    val_frac=val_frac,
                    k_neighbors=k_neighbors,
                    verbose=False,
                    weight_decay=float(wd),
                    early_stopping_patience=int(early_stopping_patience),
                    precomputed_train=precomputed_train,
                    precomputed_val=precomputed_val,
                )
                v_losses = [v for v in hist["val_loss"] if not np.isnan(v)]
                cell_best_val = float(min(v_losses)) if v_losses else float("nan")
                rec = {
                    "hidden": int(hidden),
                    "weight_decay": float(wd),
                    "best_val_mse": cell_best_val,
                    "best_epoch": int(hist["best_epoch"]),
                    "stopped_early": bool(hist["stopped_early"]),
                    "n_epochs_run": len(hist["train_loss"]),
                    "wall_time_s": float(hist["wall_time_s"]),
                }
                grid_records.append(rec)
                if verbose:
                    print(
                        f"[PAV-select] hidden={hidden:4d} wd={wd:.0e} "
                        f"val_mse={cell_best_val:.6f} best_epoch={rec['best_epoch']} "
                        f"early={rec['stopped_early']}"
                    )
                if np.isfinite(cell_best_val) and cell_best_val < best_val - 1e-12:
                    best_val = cell_best_val
                    best_cell = rec

        if best_cell is None:
            raise RuntimeError(
                "fit_with_selection found no cell with finite val MSE; "
                "grid_records=" + str(grid_records)
            )

        # Refit at the best cell with a fresh seed so the returned PAV is
        # the canonical "best" model (not just whichever sweep iterate
        # happened to be in memory).
        key, refit_key = jax.random.split(key)
        best_pav = cls(
            state_dim=int(state_dim),
            hidden=int(best_cell["hidden"]),
            dropout=float(dropout),
        )
        final_hist = best_pav.fit(
            trajectories=trajectories,
            terminal_success=terminal_success,
            n_epochs=int(n_epochs),
            lr=float(lr),
            key=refit_key,
            val_frac=val_frac,
            k_neighbors=k_neighbors,
            verbose=verbose,
            weight_decay=float(best_cell["weight_decay"]),
            early_stopping_patience=int(early_stopping_patience),
            precomputed_train=precomputed_train,
            precomputed_val=precomputed_val,
        )

        report = {
            "best_hidden": int(best_cell["hidden"]),
            "best_weight_decay": float(best_cell["weight_decay"]),
            "best_val_mse": float(best_cell["best_val_mse"]),
            "best_epoch": int(best_cell["best_epoch"]),
            "grid": grid_records,
            "total_wall_time_s": float(time.time() - t_total),
            "final_history": final_hist,
        }
        return best_pav, report

    # ---------------------------------------------------------------- score
    def score(self, trajectory: Trajectory) -> float:
        """Score a trajectory: sum of per-step predicted advantages.

        Returns 0.0 if the model has not been fit yet (so callers always
        get a finite scalar).
        """
        if self._model is None or self._state_mu is None or self._state_sd is None:
            return 0.0
        actions = np.asarray(trajectory.actions)
        H = int(actions.shape[0])
        if H <= 0:
            return 0.0
        per_step_states = np.stack(
            [_state_at_step(trajectory, k, H) for k in range(H)],
            axis=0,
        )
        x = (per_step_states - self._state_mu) / self._state_sd
        x_j = jnp.asarray(x.astype(np.float32))
        # vmap over steps with no dropout key (training=False).
        per_step_adv = jax.vmap(lambda v: self._model(v, training=False))(x_j)
        return float(jnp.sum(per_step_adv))

    def score_batch(self, trajectories: list[Trajectory]) -> np.ndarray:
        """Convenience: score a list of trajectories, returning a NumPy vec."""
        return np.array([self.score(t) for t in trajectories], dtype=np.float64)

    @property
    def is_fit(self) -> bool:
        return self._model is not None


# ---------------------------------------------------------------------------
# Loss / Adam helpers (kept private to this module).
# ---------------------------------------------------------------------------


def _mse_loss(
    model: _MLP,
    xs: jax.Array,
    ys: jax.Array,
    key: jax.Array,
    training: bool,
) -> jax.Array:
    if training and model.dropout > 0.0:
        keys = jax.random.split(key, xs.shape[0])
        preds = jax.vmap(lambda v, k: model(v, key=k, training=True))(xs, keys)
    else:
        preds = jax.vmap(lambda v: model(v, training=False))(xs)
    return jnp.mean((preds - ys) ** 2)


_loss_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(_mse_loss))


@eqx.filter_jit
def _eval_loss(
    model: _MLP,
    xs: jax.Array,
    ys: jax.Array,
) -> jax.Array:
    preds = jax.vmap(lambda v: model(v, training=False))(xs)
    return jnp.mean((preds - ys) ** 2)


def _adam_init(model: _MLP) -> dict[str, Any]:
    """Initialize Adam moments matching the model's trainable pytree."""
    params = eqx.filter(model, eqx.is_array)
    m = jax.tree.map(lambda p: jnp.zeros_like(p), params)
    v = jax.tree.map(lambda p: jnp.zeros_like(p), params)
    return {"m": m, "v": v, "t": 0}


def _adam_step(
    model: _MLP,
    grads: _MLP,
    state: dict[str, Any],
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[_MLP, dict[str, Any]]:
    """One Adam (or AdamW) step on the array leaves of the model pytree.

    When ``weight_decay > 0`` we apply *decoupled* weight decay
    (Loshchilov & Hutter 2019, AdamW). i.e. ``p <- p - lr * (m_hat /
    (sqrt(v_hat) + eps) + wd * p)``. Decay is applied uniformly to every
    array leaf (weights and biases). With ``weight_decay == 0`` this is
    plain Adam, identical to the previous behavior.
    """
    state = dict(state)
    state["t"] = int(state["t"]) + 1
    t = state["t"]

    g_arrays = eqx.filter(grads, eqx.is_array)
    new_m = jax.tree.map(
        lambda mp, g: beta1 * mp + (1.0 - beta1) * g,
        state["m"],
        g_arrays,
    )
    new_v = jax.tree.map(
        lambda vp, g: beta2 * vp + (1.0 - beta2) * (g * g),
        state["v"],
        g_arrays,
    )

    bc1 = 1.0 - beta1**t
    bc2 = 1.0 - beta2**t
    wd = float(weight_decay)

    def _upd(p: jax.Array, mp: jax.Array, vp: jax.Array) -> jax.Array:
        m_hat = mp / bc1
        v_hat = vp / bc2
        return p - lr * (m_hat / (jnp.sqrt(v_hat) + eps) + wd * p)

    params = eqx.filter(model, eqx.is_array)
    new_params = jax.tree.map(_upd, params, new_m, new_v)
    new_model = eqx.combine(new_params, eqx.filter(model, lambda x: not eqx.is_array(x)))
    state["m"] = new_m
    state["v"] = new_v
    return new_model, state


__all__ = [
    "PAVProcessRewardModel",
    "PAVTrainingHistory",
    "StepDataset",
    "compute_per_step_mc_labels",
]
