"""STL specifications for the ``cardiac_ap`` task family (FitzHugh-Nagumo).

Three specs of varying difficulty over the same FHN state ``s_t = (V, w)``:

* ``cardiac.depolarize.easy``       -- single spike (depolarisation reach).
* ``cardiac.train.medium``          -- two spikes with refractory between.
* ``cardiac.suppress_after_two.hard`` -- two spikes then sustained suppression.

State convention. The signal channels are the 2-state FHN output emitted by
:class:`stl_seed.tasks.cardiac_ap.CardiacAPSimulator`:

    y[0] = V    (membrane voltage, dimensionless)         <- read by all specs
    y[1] = w    (recovery variable, dimensionless)

The horizon is ``T = 100`` dimensionless time units, which corresponds to
~100 ms of physical time under the standard FHN cardiac time-scaling
(Aliev & Panfilov 1996 *Chaos Solitons Fractals* 7:293). At the canonical
parameters ``(a, b, epsilon) = (0.7, 0.8, 0.08)`` the inter-spike interval
under sustained suprathreshold stimulation is ~40 time units (FitzHugh
1961 §III, Fig. 1), so 100 time units fits roughly two spike cycles --
the natural granularity for the three specs below.

Action. ``u_t in [0, 1]`` is a fractional external membrane current; the
simulator passes it through unchanged into ``I_ext(t)``. ``H = 10``
control updates over 100 time units (one update every 10 units, longer
than the ~3-unit fast V relaxation but much shorter than the ~12-unit
refractory window).

Phase-portrait threshold derivation (FitzHugh 1961 §III; Murray
*Mathematical Biology I* §1.5; Keener & Sneyd *Mathematical
Physiology* Ch. 5):

* The cubic V-nullcline ``w = V - V**3 / 3`` has a local maximum at
  ``V = +1`` (where ``w = +2/3``) and a local minimum at ``V = -1``
  (where ``w = -2/3``). The interval ``V in [-1, +1]`` is the
  "negative-slope branch" inside which trajectories are repelled away
  from the V-nullcline (the auto-catalytic portion); ``V > +1`` is
  the depolarised branch and ``V < -1`` is the resting branch.

* Threshold for firing: a perturbation that pushes ``V`` above the
  local maximum at ``V = +1`` triggers the spike-and-recovery cycle;
  a perturbation that does not is reabsorbed. We therefore use
  ``V_PEAK = +1.0`` as the "depolarised" threshold for the spike
  predicates (FitzHugh 1961 Fig. 1; Murray Fig. 1.6). The spike peak
  reaches V ~ +2.0 under sustained stimulation, so ``V > 1.0``
  cleanly discriminates "fired" from "did not fire".

* The autonomous resting fixed point of the FHN system at the default
  parameters is at ``V* ~ -1.20``, w* ~ -0.625`` (numerical solution
  of the cubic intersection). The cell is "back at rest" when
  ``V`` returns to a band around this value; we use ``V_REST = +0.5``
  as the suppression threshold (well above ``V*`` but well below the
  ``V_PEAK = +1.0`` firing threshold, so any trajectory with
  ``V < +0.5`` is unambiguously NOT firing). This is the textbook
  "subthreshold" band per FitzHugh 1961 §III.

Conjunction-only structure (firewall §C.1): every formula in this module is
assembled from ``Always``, ``Eventually``, n-ary ``And``, and predicate-level
``Negation``. No top-level disjunction, no implication, no ``Until``.

dimensionless thresholds ``{0.50, 0.30, 0.90, 0.45, 0.35, 0.80, 0.40, 0.70,
1.5, 0.20, 0.60}`` on signals ``x_1..x_4 in [0, 1.5]`` over ``t in [0, 25]``.
The thresholds in this module are FHN dimensionless voltages
``{V_PEAK = 1.0, V_REST = 0.5}`` over ``t in [0, 100]`` on the 2-state
``(V, w)`` system. The ``V_REST = 0.5`` literal coincides numerically with
subthreshold band) and the *signal scale* (V ranges over [-2, +2.5] over
incidental rather than transcribed (firewall §D.3 protocol).
"""

from __future__ import annotations

from stl_seed.specs import (
    Always,
    And,
    Eventually,
    Interval,
    Predicate,
    STLSpec,
    register,
)

# ---------------------------------------------------------------------------
# Predicate factories (channel/threshold parametrised, mirroring the bio_ode
# convention).
# ---------------------------------------------------------------------------


def _gt(name: str, channel: int, threshold: float) -> Predicate:
    """``signal[channel] - threshold`` (predicate-level robustness)."""

    return Predicate(
        f"{name}>{threshold}",
        fn=lambda traj, t, c=channel, th=threshold: float(traj[t, c]) - th,
    )


def _lt(name: str, channel: int, threshold: float) -> Predicate:
    """``threshold - signal[channel]`` (predicate-level robustness)."""

    return Predicate(
        f"{name}<{threshold}",
        fn=lambda traj, t, c=channel, th=threshold: th - float(traj[t, c]),
    )


# ---------------------------------------------------------------------------
# Phase-portrait-derived thresholds (single source of truth).
# ---------------------------------------------------------------------------

#: Voltage threshold marking "depolarised" / "fired" cell. Derived from the
#: FHN cubic V-nullcline ``w = V - V**3 / 3`` whose local maximum sits at
#: ``V = +1`` (FitzHugh 1961 §III; Murray *Mathematical Biology I* §1.5
#: Fig. 1.6). Trajectories with ``V > 1.0`` are unambiguously on the
#: depolarised branch.
V_PEAK = 1.0

#: Voltage threshold marking "back to rest" / "subthreshold". Sits well
#: above the resting fixed point ``V* ~ -1.20`` but well below the firing
#: threshold ``V_PEAK = 1.0``, so ``V < 0.5`` is unambiguously NOT firing
#: (the "subthreshold band" per FitzHugh 1961 §III).
V_REST = 0.5

#: Total simulation horizon (dimensionless FHN time units). Matches
#: :data:`stl_seed.tasks.cardiac_ap.CARDIAC_HORIZON_TU` exactly.
CARDIAC_T = 100.0


# ---------------------------------------------------------------------------
# Spec 1 (EASY): cardiac.depolarize.easy -- fire at least once in first half.
# ---------------------------------------------------------------------------
#
# Textbook source. FitzHugh 1961 §III defines the FHN excitability
# threshold at the local maximum of the cubic V-nullcline (V = +1); a
# trajectory that crosses this threshold has fired an action potential.
# The "fire at least once" requirement is the minimal excitability test:
# can the controller produce *any* suprathreshold drive in the first
# half of the horizon?
#
# Difficulty: EASY. Even a constant-on policy ``u = 1`` saturates the
# external current and triggers a spike within the first ~10 time units
# (the FHN spike rise time at default params is ~3 units, well inside
# the 50-unit window). Random policies satisfy the spec a meaningful
# fraction of the time, putting random-policy success rate in the
# textbook calibration band per the firewall §C.1 conjunction-only
# allowed form.

depolarize_spec = STLSpec(
    name="cardiac.depolarize.easy",
    formula=Eventually(
        _gt("V_fires", 0, V_PEAK),
        interval=Interval(0.0, 50.0),
    ),
    signal_dim=2,
    horizon_minutes=CARDIAC_T,  # repurposed: dimensionless time units (FHN convention)
    description=(
        "Fire at least one action potential (V crosses the cubic-V-nullcline "
        "local maximum at V = 1.0) within the first half of the horizon "
        "[0, 50] time units. Form: single Eventually clause."
    ),
    citations=(
        "FitzHugh, R. Biophys J 1(6):445-466 (1961), DOI 10.1016/S0006-3495(61)86902-6, §III.",
        "Nagumo, J., Arimoto, S., Yoshizawa, S. Proc IRE 50(10):2061-2070 "
        "(1962), DOI 10.1109/JRPROC.1962.288235, Eq. 4.",
        "Murray, J.D. Mathematical Biology I (3rd ed., 2002), §1.5, Fig. 1.6.",
    ),
    formula_text="F_[0,50] (V > 1.0)",
    metadata={
        "subdomain": "cardiac",
        "difficulty": "easy",
        "horizon_tu": CARDIAC_T,
        "control_points": 10,
        "thresholds_dimensionless": {"V_PEAK": V_PEAK},
        "allowed_form": "single reachability clause (firewall §C.1)",
    },
)
register(depolarize_spec)


# ---------------------------------------------------------------------------
# Spec 2 (MEDIUM): cardiac.train.medium -- fire twice with refractory between.
# ---------------------------------------------------------------------------
#
# Textbook source. FitzHugh 1961 §III reports an inter-spike interval of
# ~40 time units under sustained suprathreshold stimulation; the
# refractory window after a spike is on the order of ``1/epsilon ~ 12.5``
# time units (the slow-w time-scale). The "fire twice" requirement
# therefore demands the controller produce *two* spike events separated
# by enough time for the recovery variable ``w`` to relax back below
# the firing threshold -- a non-trivial timing constraint.
#
# Difficulty: MEDIUM. A constant-on policy yields sustained periodic
# spiking and trivially satisfies the spec; a constant-off policy fires
# zero spikes and fails. The interesting failure mode is a policy that
# fires once early but then either does not relax (if u stays high) or
# does not re-fire (if u drops too low). Random policies hit either the
# satisfaction band or one of these failure modes with roughly equal
# probability.
#
# Spec form. The two-spike requirement is encoded as a conjunction of two
# Eventually clauses on disjoint time windows:
#
#   F_[0,30]  (V > 1.0)   -- first spike in the early window
#   F_[40,70] (V > 1.0)   -- second spike after the refractory gap
#
# The 10-unit gap between [0,30] and [40,70] forces the controller to
# allow the FHN ``w`` recovery (which decays with time-constant
# ``1/(epsilon * b) = 1 / (0.08 * 0.8) ~ 15.6`` time units) to bring
# the cell back below threshold before the second spike, matching the
# textbook refractory-period structure.

train_spec = STLSpec(
    name="cardiac.train.medium",
    formula=And(
        children=(
            Eventually(
                _gt("V_first_spike", 0, V_PEAK),
                interval=Interval(0.0, 30.0),
            ),
            Eventually(
                _gt("V_second_spike", 0, V_PEAK),
                interval=Interval(40.0, 70.0),
            ),
        )
    ),
    signal_dim=2,
    horizon_minutes=CARDIAC_T,
    description=(
        "Fire two action potentials with the FHN refractory window "
        "between them: first spike in [0, 30] time units, second spike in "
        "[40, 70] time units. The 10-unit gap forces the slow recovery "
        "variable w (time constant 1/(epsilon*b) ~ 15.6 units) to relax "
        "below threshold before the second spike. Form: conjunction of two "
        "reachability clauses on disjoint windows."
    ),
    citations=(
        "FitzHugh, R. Biophys J 1(6):445-466 (1961), DOI 10.1016/S0006-3495(61)86902-6, "
        "§III, Fig. 1 (inter-spike interval ~40 time units).",
        "Nagumo, J., Arimoto, S., Yoshizawa, S. Proc IRE 50(10):2061-2070 (1962), Eq. 4.",
        "Keener, J. & Sneyd, J. Mathematical Physiology (2nd ed., 2009), Ch. 5 "
        "(refractory period in excitable membranes).",
    ),
    formula_text="F_[0,30] (V > 1.0) AND F_[40,70] (V > 1.0)",
    metadata={
        "subdomain": "cardiac",
        "difficulty": "medium",
        "horizon_tu": CARDIAC_T,
        "control_points": 10,
        "thresholds_dimensionless": {"V_PEAK": V_PEAK},
        "allowed_form": "conjunction of two reachability clauses (firewall §C.1)",
    },
)
register(train_spec)


# ---------------------------------------------------------------------------
# Spec 3 (HARD): cardiac.suppress_after_two.hard -- two spikes then suppress.
# ---------------------------------------------------------------------------
#
# Textbook source. The "fire twice then suppress" pattern is the canonical
# pacing-then-quiescence test for a cardiac excitable membrane: deliver a
# pacing protocol that elicits exactly two spikes, then withdraw drive and
# verify the cell returns to (and stays at) the resting state. This is the
# textbook test for an antiarrhythmic intervention (Keener & Sneyd
# *Mathematical Physiology* Ch. 5; Aliev & Panfilov 1996 *Chaos Solitons
# Fractals* 7:293, "Simple two-variable model of cardiac excitation").
#
# Difficulty: HARD. The spec demands (i) a first spike in [0, 30],
# (ii) a second spike in [40, 60], (iii) sustained subthreshold
# behaviour ``V < 0.5`` over the entire window [70, 100]. The third
# clause is the binding constraint: once the cell has fired twice, the
# controller must reduce ``I_ext`` enough to let ``w`` re-equilibrate
# AND avoid re-firing. A random policy that satisfies (i) and (ii) is
# overwhelmingly likely to either over-stimulate (failing iii by
# triggering a third spike in [70, 100]) or under-stimulate (failing
# (ii) by not firing the second spike). The sweet spot is a brief
# pulse-train that delivers exactly two spikes and then drops to zero
# current, matching the textbook "burst-and-quiet" intervention pattern.
#
# Spec form. Conjunction of two Eventually clauses (the two spikes) and
# one Always clause (the sustained suppression):
#
#   F_[0,30]    (V > 1.0)   -- first spike
#   F_[40,60]   (V > 1.0)   -- second spike (tighter window than the
#                              medium spec to force tighter timing)
#   G_[70,100]  (V < 0.5)   -- stay subthreshold over the back third

suppress_after_two_spec = STLSpec(
    name="cardiac.suppress_after_two.hard",
    formula=And(
        children=(
            Eventually(
                _gt("V_first_spike", 0, V_PEAK),
                interval=Interval(0.0, 30.0),
            ),
            Eventually(
                _gt("V_second_spike", 0, V_PEAK),
                interval=Interval(40.0, 60.0),
            ),
            Always(
                _lt("V_suppressed", 0, V_REST),
                interval=Interval(70.0, CARDIAC_T),
            ),
        )
    ),
    signal_dim=2,
    horizon_minutes=CARDIAC_T,
    description=(
        "Pace then quiesce: two spikes (V > 1.0 in [0, 30] and [40, 60]) "
        "followed by sustained subthreshold behaviour (V < 0.5 throughout "
        "[70, 100]). Encodes the textbook pacing-then-antiarrhythmic-"
        "withdrawal test for an excitable cardiac membrane. Form: "
        "conjunction of two reachability + one tracking clause."
    ),
    citations=(
        "FitzHugh, R. Biophys J 1(6):445-466 (1961), §III.",
        "Nagumo, J. et al. Proc IRE 50(10):2061-2070 (1962), Eq. 4.",
        "Keener, J. & Sneyd, J. Mathematical Physiology (2nd ed., 2009), Ch. 5 "
        "(pacing and quiescence in excitable membranes).",
        "Aliev, R.R. & Panfilov, A.V. Chaos Solitons Fractals 7:293 (1996), "
        "DOI 10.1016/0960-0779(95)00089-5 (cardiac FHN time-scaling).",
    ),
    formula_text=("F_[0,30] (V > 1.0) AND F_[40,60] (V > 1.0) AND G_[70,100] (V < 0.5)"),
    metadata={
        "subdomain": "cardiac",
        "difficulty": "hard",
        "horizon_tu": CARDIAC_T,
        "control_points": 10,
        "thresholds_dimensionless": {
            "V_PEAK": V_PEAK,
            "V_REST": V_REST,
        },
        "allowed_form": ("conjunction of two reachability + one tracking clause (firewall §C.1)"),
    },
)
register(suppress_after_two_spec)


__all__ = [
    "depolarize_spec",
    "train_spec",
    "suppress_after_two_spec",
    "V_PEAK",
    "V_REST",
    "CARDIAC_T",
]
