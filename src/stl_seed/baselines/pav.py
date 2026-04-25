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

REDACTED firewall
-------------
This module imports only from JAX, Equinox, and the in-repo
:class:`stl_seed.tasks._trajectory.Trajectory` type. No REDACTED,
REDACTED, REDACTED, REDACTED, or REDACTED artifacts.

Architecture
------------
The MLP is a 3-layer Equinox feed-forward network:

    Linear(state_dim -> hidden) -> ReLU -> Dropout(p)
    Linear(hidden     -> hidden) -> ReLU -> Dropout(p)
    Linear(hidden     -> 1)              # scalar advantage prediction

with hidden=256, dropout=0.1 by default. Equinox is consistent with the
rest of the JAX-native stack; we deliberately avoid PyTorch.

Per-step MC labels
------------------
Setlur et al. label step t with the *advantage*:

    A_t  =  P_succeed(state after step t)  -  P_succeed(state before step t)

P_succeed at a state is the empirical fraction of *continuations* from that
state that succeed at the terminal. With a finite trajectory store and no
ability to draw fresh continuations, we estimate P_succeed at state s_t via
nearest-neighbor pooling: the empirical success fraction among the K most
similar (state, step) pairs in the corpus (Manhattan distance in the
post-z-scored state space). This is a standard substitute for fresh
continuations when the per-step distribution is dense enough that K
neighbors are well-defined and the verifier need only be approximately
calibrated. We document this clearly so a reviewer can attack it: it is the
empirical-PAV variant, not the on-policy-rollout PAV. The substitute is
what allows PAV to be trained at zero additional simulation cost on the
same canonical store STL-rho consumes.

Training
--------
``fit(...)`` minimizes mean-squared error between predicted and target
per-step advantages, with per-trajectory random partitions used to mint
the (state, advantage) training pairs. The loss history is returned so the
caller can validate convergence. Adam optimizer (lr=1e-3 default), full-
batch by default for stability on the small (typically O(10k)) (state,
advantage) sets we work with.
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

        Returns
        -------
        dict with keys:
            ``train_loss``: list[float], one per epoch (MSE).
            ``val_loss``: list[float], one per epoch.
            ``wall_time_s``: float, seconds for the entire fit call.
            ``n_train_pairs``: int.
            ``n_val_pairs``: int.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        ts = np.asarray(terminal_success, dtype=np.float64)
        n_traj = len(trajectories)
        if n_traj < 2:
            raise ValueError(f"PAV.fit needs >=2 trajectories to form a kNN pool; got {n_traj}")

        # Trajectory-level train/val split (preserves leave-one-out semantics
        # of the kNN estimator within each side).
        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
        perm = rng.permutation(n_traj)
        n_val = max(1, int(round(val_frac * n_traj))) if val_frac > 0.0 else 0
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if len(tr_idx) < 2:
            # Not enough train trajectories — collapse split so kNN works.
            tr_idx = perm
            val_idx = np.array([], dtype=np.int64)

        train_trajs = [trajectories[int(i)] for i in tr_idx]
        train_ts = ts[tr_idx]
        val_trajs = [trajectories[int(i)] for i in val_idx]
        val_ts = ts[val_idx]

        train_ds = compute_per_step_mc_labels(train_trajs, train_ts, k_neighbors=k_neighbors)
        val_ds: StepDataset | None = None
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

        # Hand-rolled Adam to avoid taking on optax as a dependency
        # (it's a separate package and not in pyproject.toml).
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
                model, opt_state = _adam_step(model, grads, opt_state, lr=float(lr))
                epoch_losses.append(float(loss))
            train_loss = float(np.mean(epoch_losses))
            history.train_loss.append(train_loss)

            if val_x_j is not None:
                v_loss = float(_eval_loss(model, val_x_j, val_y_j))
                history.val_loss.append(v_loss)
            else:
                history.val_loss.append(float("nan"))

            if verbose:
                print(
                    f"[PAV] epoch {epoch + 1}/{n_epochs} "
                    f"train_mse={train_loss:.6f} val_mse={history.val_loss[-1]:.6f}"
                )

        history.wall_time_s = float(time.time() - t0)
        self._model = model
        return {
            "train_loss": list(history.train_loss),
            "val_loss": list(history.val_loss),
            "wall_time_s": float(history.wall_time_s),
            "n_train_pairs": int(history.n_train_pairs),
            "n_val_pairs": int(history.n_val_pairs),
        }

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
) -> tuple[_MLP, dict[str, Any]]:
    """One Adam step. Updates only the array leaves of the model pytree."""
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

    def _upd(p: jax.Array, mp: jax.Array, vp: jax.Array) -> jax.Array:
        m_hat = mp / bc1
        v_hat = vp / bc2
        return p - lr * m_hat / (jnp.sqrt(v_hat) + eps)

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
