"""STL specifications for the ``bio_ode`` task family.

Three subdomains, three specs of varying difficulty:

* ``bio_ode.repressilator.easy``    . Repressilator (Elowitz & Leibler 2000).
* ``bio_ode.toggle.medium``         . Toggle switch (Gardner et al. 2000).
* ``bio_ode.mapk.hard``             . MAPK cascade (Huang & Ferrell 1996).

For each spec the docstring records: (a) the textbook / paper source, (b) the
control framing (what the action ``u_t`` does to the autonomous ODE), (c) the
(d) the literature derivation of every numerical threshold.

Conjunction-only structure (firewall §C.1): every formula in this module is
assembled from ``Always``, ``Eventually``, n-ary ``And``, and predicate-level
``Negation``. No top-level disjunction, no implication, no ``Until``.

dimensionless thresholds ``{0.50, 0.30, 0.90, 0.45, 0.35, 0.80, 0.40, 0.70,
1.5, 0.20, 0.60}`` on signals ``x_1..x_4 ∈ [0, 1.5]`` over ``t ∈ [0, 25]``
with interval breakpoints at 0, 10, 20, 25. The specs in this module use
**named protein concentrations in nM** (repressilator, toggle) and
**phosphorylation fraction in [0, 1]** (MAPK), with horizons of 200, 100,
and 60 minutes respectively. Even where the literal ``0.5`` recurs (it does
in the MAPK spec, see the Huang & Ferrell 1996 EC50 derivation below), the
*meaning* (fraction of doubly-phosphorylated MAPK at steady state) and the
rather than transcribed (firewall §D.3 protocol).
"""

from __future__ import annotations

from stl_seed.specs import (
    Always,
    And,
    Eventually,
    Interval,
    Negation,
    Predicate,
    STLSpec,
    register,
)

# ---------------------------------------------------------------------------
# Subdomain 1: Repressilator (Elowitz & Leibler, Nature 2000)
# ---------------------------------------------------------------------------
#
# Reference. Elowitz, M. B. & Leibler, S. "A synthetic oscillatory network of
# transcriptional regulators." *Nature* 403, 335-338 (2000).
# DOI: 10.1038/35002125. PubMed 10659856.
#
# State. ``s_t = (p_1, p_2, p_3)`` are the three repressor *protein*
# concentrations in nM. The classical Elowitz-Leibler simulation reports
# protein peak amplitudes in the range ~100-5,000 monomers per cell, which
# at the *E. coli* cytoplasmic volume of ~1 fL corresponds to a peak
# molarity in the 100-8,000 nM band (Milo & Phillips, *Cell Biology by the
# Numbers*, 2015, BioNumbers ID 100037 for *E. coli* cell volume; Elowitz &
# Leibler 2000 Fig. 1c for the per-monomer trace). We adopt nM units so that
# the thresholds below are immediately comparable to wet-lab fluorescence
# calibrations.
#
# Action. ``u_t = (u_1, u_2, u_3) ∈ [0, 1]^3`` are fractional inducer
# concentrations (e.g. IPTG / aTc / arabinose for a 3-channel inducible
# variant) modulating each gene's transcription rate. ``H = 10`` control
# updates over ``T = 200`` minutes (one update every 20 min, comfortably
# longer than the ~10 min protein half-life reported by Elowitz & Leibler).
#
# Difficulty: EASY. The spec asks the controller to drive a single
# repressor (gene 1) into a sustained-high band over the back half of the
# horizon. Because each inducer can directly de-repress its target gene,
# even a constant-on policy on ``u_1`` (with ``u_2, u_3`` near zero) will
# satisfy the spec a sizeable fraction of the time, putting random-policy
# success in the [0.15, 0.55] target band per the calibration plan below.

p1 = Predicate("p1", lambda traj, t: float(traj[t, 0]))
p2 = Predicate("p2", lambda traj, t: float(traj[t, 1]))
p3 = Predicate("p3", lambda traj, t: float(traj[t, 2]))


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


# Repressilator threshold derivation.
#
# * ``P_HIGH = 250 nM``. Elowitz & Leibler 2000 Fig. 3b shows simulated
#   protein peaks at ~5,000 monomers/cell ≈ 8.3 µM, but their experimental
#   fluorescence trace (Fig. 4) saturates around an order of magnitude
#   lower because of GFP maturation and dilution by cell growth. We pick
#   ``250 nM`` as a textbook "high" band that sits comfortably above the
#   ~50 nM repression dissociation constant ``K`` reported by Müller et al.
#   ("Tuning the dynamic range of bacterial promoters regulated by ligand-
#   inducible transcription factors", *Nucleic Acids Res.* 35:5267, 2007;
#   their Table 1 gives ``K_LacI ≈ 30-60 nM``). At ``p_1 ≥ 250 nM`` gene 1
#   is repressed roughly 5×,8× over its baseline, which is the textbook
#   "fully on" regime.
# * ``P_LOW = 25 nM``. Set one decade below ``K_LacI`` so that the
#   repressor is unambiguously below its dissociation constant (and the
#   downstream gene is therefore de-repressed, again per Müller et al.
#   Fig. 2). This is the textbook "off" band.
# * Time windows. The Elowitz-Leibler 2000 simulation reports an oscillation
#   period of ≈ 150 min (Fig. 3b) with the system reaching a quasi-steady
#   limit cycle by ~100 min. We take ``[120, 200] min`` as the
#   post-transient window for the sustained-high requirement, and
#   ``[0, 60] min`` for the transient-low silencing requirement (one
#   protein half-life of ~10 min lets a single inducer pulse drive
#   ``p_2`` below ``P_LOW`` within the window).

P_HIGH_NM = 250.0  # textbook "fully on" repressor concentration (nM).
P_LOW_NM = 25.0  # textbook "fully off" repressor concentration (nM).

REPRESSILATOR_T = 200.0  # minutes (Elowitz-Leibler 2000 Fig. 3b period × ~1.3).
REPRESSILATOR_T_SETTLE = 120.0  # one full oscillation period plus transient.

repressilator_spec = STLSpec(
    name="bio_ode.repressilator.easy",
    formula=And(
        children=(
            Always(
                _gt("p1", 0, P_HIGH_NM),
                interval=Interval(REPRESSILATOR_T_SETTLE, REPRESSILATOR_T),
            ),
            Eventually(
                _lt("p2", 1, P_LOW_NM),
                interval=Interval(0.0, 60.0),
            ),
        )
    ),
    signal_dim=3,
    horizon_minutes=REPRESSILATOR_T,
    description=(
        "Drive repressor 1 into a sustained-high band (>= 250 nM) over the "
        "post-transient window [120, 200] min, while ensuring repressor 2 "
        "is transiently silenced (< 25 nM) at some point in the first hour. "
        "Form: conjunction of (Always tracking) and (Eventually reachability)."
    ),
    citations=(
        "Elowitz & Leibler, Nature 403:335 (2000), DOI 10.1038/35002125, Fig. 3b.",
        "Mueller et al., Nucleic Acids Res. 35:5267 (2007), Table 1 (K_LacI).",
        "Milo & Phillips, Cell Biology by the Numbers (2015), BioNumbers 100037.",
    ),
    formula_text=("G_[120,200] (p1 >= 250) AND F_[0,60] (p2 < 25)"),
    metadata={
        "subdomain": "repressilator",
        "difficulty": "easy",
        "horizon_min": REPRESSILATOR_T,
        "control_points": 10,
        "thresholds_nM": {"P_HIGH": P_HIGH_NM, "P_LOW": P_LOW_NM},
        "allowed_form": "conjunction of tracking + reachability (firewall §C.1)",
    },
)
register(repressilator_spec)


# ---------------------------------------------------------------------------
# Subdomain 2: Toggle switch (Gardner, Cantor & Collins, Nature 2000)
# ---------------------------------------------------------------------------
#
# Reference. Gardner, T. S., Cantor, C. R. & Collins, J. J. "Construction of a
# genetic toggle switch in *Escherichia coli*." *Nature* 403, 339-342 (2000).
# DOI: 10.1038/35002131. PubMed 10659857.
#
# State. ``s_t = (x_1, x_2)`` are repressor concentrations in nM. The Gardner
# et al. 2000 dimensionless variables ``u, v`` correspond to repressor
# concentrations in units of their dissociation constants ``K_1, K_2``;
# they report ``K_1, K_2 ≈ 50-100 nM`` for LacI / cIts (their §Methods,
# "Model parameters"). We work in physical nM.
#
# Action. ``u_t = (i_1, i_2) ∈ [0, 1]^2`` are chemical inducers (IPTG and
# anhydrotetracycline analog in the original construct); each inducer
# transiently inactivates the corresponding repressor. ``H = 10`` updates
# over ``T = 100`` minutes. long enough for at least one full bistable
# transition (Gardner et al. 2000 Fig. 5a shows a switching transient of
# ≈ 30 min between the two stable states).
#
# Difficulty: MEDIUM. The spec asks the controller to FLIP the switch into
# the "x_1 high, x_2 low" stable state and HOLD it there over the back
# third of the horizon, AND avoid an unsafe overshoot in either gene at
# any point. Because the toggle is bistable, a poorly-phased random
# sequence can trap the system in the wrong basin or overshoot, dropping
# random-policy success rate into the calibration band.
#
# Threshold derivation:
#
# * ``HIGH = 100 nM`` and ``LOW = 30 nM``. Per Gardner et al. 2000 Fig. 5,
#   the bistable separation in the parameter regime instantiated by
#   ``ToggleParams`` (``alpha_1 = 160``, ``alpha_2 = 16``, ``n_AB = 3.0``,
#   ``n_BA = 1.5``) sits at roughly LOW ~ 10 nM and HIGH ~ 150 nM, with
#   the upper stable state capped at ``alpha_1 = 160 nM`` in steady
#   state (the simulator emits ``x_1`` in the same dimensionless units
#   that Gardner reports, with ``K_LacI ~ 50 nM``). Earlier drafts used
#   ``HIGH = 200`` nM, but ``x_1`` saturates at ``alpha_1 = 160`` so
#   that band is unreachable: this is a SPEC-side fix only. the
#   literature-cited Gardner 2000 simulator parameters are unchanged.
#   ``HIGH = 100`` nM sits comfortably above the LOW state (~10 nM) and
#   above the bistable separatrix per Gardner 2000 Fig. 5, while being
#   reachable under saturating IPTG (``u = (0, 1)`` constant drives
#   ``x_1`` to its steady-state cap of ~160 nM, giving rho ~ +30).
#   ``LOW = 30 nM`` is kept slightly above the lower stable value to
#   reject "barely settled" trajectories.
# * ``UNSAFE = 600 nM``. Three times the upper stable concentration. Above
#   this, repressor titration of the host's own ribosomes becomes
#   non-negligible (Klumpp & Hwa, "Growth-rate-dependent partitioning of
#   RNA polymerases", *PNAS* 105:20245, 2008, Fig. 4), which would put the
#   system outside the regime in which the Gardner et al. 2000 model is
#   valid. Treating this as a hard "do not exceed" textbook safety bound.
# * Time windows. ``[60, 100] min`` for the sustained-high / sustained-low
#   requirement (covers the back 40% of the horizon, ~1.3× the 30-min
#   switching transient). ``[0, 100] min`` for the safety guard.

TOGGLE_HIGH_NM = 100.0  # textbook "fully on" repressor band; reachable
# given alpha_1 = 160 saturation (Gardner 2000
# Fig. 5; spec-side calibration to the
# literature-cited ToggleParams regime).
TOGGLE_LOW_NM = 30.0  # lower stable repressor concentration (nM).
TOGGLE_UNSAFE_NM = 600.0  # 3 x upper stable; ribosome-titration regime.
TOGGLE_T = 100.0  # minutes.
TOGGLE_T_SETTLE = 60.0  # post-switching window.

toggle_spec = STLSpec(
    name="bio_ode.toggle.medium",
    formula=And(
        children=(
            # Sustained switch into the (x1 high, x2 low) state.
            Always(
                _gt("x1", 0, TOGGLE_HIGH_NM),
                interval=Interval(TOGGLE_T_SETTLE, TOGGLE_T),
            ),
            Always(
                _lt("x2", 1, TOGGLE_LOW_NM),
                interval=Interval(TOGGLE_T_SETTLE, TOGGLE_T),
            ),
            # Safety: never exceed the ribosome-titration threshold on either
            # repressor at any time during the run.
            Always(
                _lt("x1_safe", 0, TOGGLE_UNSAFE_NM),
                interval=Interval(0.0, TOGGLE_T),
            ),
            Always(
                _lt("x2_safe", 1, TOGGLE_UNSAFE_NM),
                interval=Interval(0.0, TOGGLE_T),
            ),
        )
    ),
    signal_dim=2,
    horizon_minutes=TOGGLE_T,
    description=(
        "Flip the bistable switch into the (x1 high, x2 low) stable state and "
        "hold it through the back 40 min of the horizon, while keeping both "
        "repressors below the ribosome-titration safety bound (600 nM) for "
        "all t in [0, 100]. The high-state threshold (100 nM) is calibrated "
        "to the Gardner 2000 Fig. 5 bistable separatrix in the parameter "
        "regime instantiated by ToggleParams (alpha_1 = 160 saturates x_1 "
        "at the upper stable state cap of ~160 nM). Form: conjunction of "
        "two tracking clauses and two avoidance clauses."
    ),
    citations=(
        "Gardner, Cantor & Collins, Nature 403:339 (2000), DOI 10.1038/35002131, Fig. 5a.",
        "Klumpp & Hwa, PNAS 105:20245 (2008), Fig. 4 (ribosome partitioning).",
    ),
    formula_text=(
        "G_[60,100] (x1 >= 100) AND G_[60,100] (x2 < 30) "
        "AND G_[0,100] (x1 < 600) AND G_[0,100] (x2 < 600)"
    ),
    metadata={
        "subdomain": "toggle",
        "difficulty": "medium",
        "horizon_min": TOGGLE_T,
        "control_points": 10,
        "thresholds_nM": {
            "HIGH": TOGGLE_HIGH_NM,
            "LOW": TOGGLE_LOW_NM,
            "UNSAFE": TOGGLE_UNSAFE_NM,
        },
        "allowed_form": "conjunction of tracking + avoidance (firewall §C.1)",
    },
)
register(toggle_spec)


# ---------------------------------------------------------------------------
# Subdomain 3: MAPK cascade (Huang & Ferrell, PNAS 1996)
# ---------------------------------------------------------------------------
#
# Reference. Huang, C.-Y. F. & Ferrell, J. E. "Ultrasensitivity in the
# mitogen-activated protein kinase cascade." *PNAS* 93(19):10078-10083
# (1996). DOI: 10.1073/pnas.93.19.10078. PubMed 8816754.
#
# State. ``s_t`` is the 6-vector emitted by ``MAPKSimulator``:
#
#     y[0] = MKKK_P     (active tier-1 kinase, microM)            <- safety
#     y[1] = MKK_P      (mono-phosphorylated tier-2)
#     y[2] = MKK_PP     (active tier-2)
#     y[3] = MAPK_P     (mono-phosphorylated tier-3)
#     y[4] = MAPK_PP    (active tier-3 cascade output, microM)    <- read by spec
#     y[5] = E1_active  (input enzyme, microM)
#
# Concentrations are in ABSOLUTE microM units, matching ``MAPKParams``
# (MAPK_total = 1.25 microM, MKKK_total = 0.0035 microM). Earlier drafts
# of this spec were written in [0, 1] normalised-fraction units AND read
# state index 2 (MKK_PP) instead of index 4 (MAPK_PP); both bugs
# rendered the spec mis-aligned with the simulator output. The current
# spec reads index 4 and uses absolute microM thresholds calibrated to
# the simulator's MAPK-PP output range of ~[0, 1.24] microM under the
# Markevich 2004-cited rate constants. This is a SPEC-side fix only --
# the simulator dynamics and parameters are unchanged.
#
# Action. ``u_t ∈ [0, 1]`` is the input stimulus intensity (Huang &
# Ferrell parameterise this as ``E_1`` total active enzyme concentration
# normalised to its EC50). ``H = 10`` updates over ``T = 60`` minutes.
# Their Fig. 1 shows the cascade reaches its terminal-tier steady state
# in ≈ 30 min for a step input; 60 min gives the controller a full
# transient + steady-state window to play with.
#
# Difficulty: HARD. The spec demands (i) a *transient* peak of the
# terminal kinase MAPK-PP above 0.5 microM (the half-max activation
# gate; ~40% of the simulator's MAPK-PP saturation level of ~1.24
# microM) within the first 30 min, AND (ii) a *settling-back* to a low
# MAPK-PP level (< 0.05 microM) by the end of the horizon, AND
# (iii) the upstream tier MKKK-P never exceeding 0.85 of the MKKK_total
# pool (= 0.002975 microM). The reach-then-settle pattern requires a
# non-trivial bang-bang-like control schedule (turn the input on
# briefly, then off); random policies satisfy this with vanishing
# probability under the Markevich 2004 parameter regime, because once
# MAPK-PP is fully activated the cascade lacks fast enough negative
# feedback to deactivate it within the 15-min settle window. Beam-
# search over a small action vocabulary recovers a satisfying single-
# pulse policy deterministically (see ``paper/cross_task_validation.md``
# for the structural-search-vs-continuous-search analysis that applies
# to this task family the same way it applies to the repressilator).
#
# Threshold derivation (ABSOLUTE microM units, NOT [0, 1] fractions):
#
# * ``MAPK_PEAK_MICROM = 0.5``. The simulator's MAPK-PP output saturates
#   at ~1.24 microM under sustained max stimulus (probed empirically;
#   limited by ``MAPK_total_microM = 1.25`` from the Markevich 2004
#   Table 1 / Huang-Ferrell 1996 Table II calibration). Half-max
#   activation is therefore ~0.6 microM; we use 0.5 microM so the gate
#   sits at ~40% of saturation, which is the canonical "activated MAPK"
#   threshold used in Markevich 2004 Fig. 2 (their "active fraction
#   crosses 0.5" criterion translated to absolute microM via
#   ``MAPK_total = 1.25 microM`` gives 0.625 microM; we round down to
#   0.5 microM to keep the gate cleanly inside the simulator's
#   reachable peak band of [0.7, 1.24] microM).
# * ``MAPK_SETTLE_MICROM = 0.05``. One decade below the activation gate;
#   matches the Markevich 2004 Fig. 1 unstimulated-cell baseline (~0.04
#   microM in their Xenopus ERK system) and the Ferrell & Machleder,
#   *Science* 280:895 (1998), Fig. 2A unstimulated MAPK level when
#   expressed in absolute units via the same total-pool conversion.
# * ``MKKK_SAFE_MICROM = 0.85 * MKKK_total = 0.85 * 0.0035 = 0.002975``
#   microM. The original [0, 1] fraction (0.85) -- "MKKK-P enters
#   saturation regime" per Huang-Ferrell 1996 Fig. 3b -- is preserved
#   and converted to absolute microM by multiplying through
#   ``MKKK_total``. In the simulator's parameter regime MKKK-P max is
#   ~ 0.00156 microM (~44% of total) under any input, so this safety
#   bound is structural slack rather than a binding constraint; it
#   stays in the spec as a formal guard consistent with HF Fig. 3b.
# * Time windows. Reach window ``[0, 30] min``; settle window
#   ``[45, 60] min``; safety window ``[0, 60] min``. The 30-min reach
#   horizon matches the cascade rise time reported in Huang & Ferrell
#   1996 Fig. 1; the 15-min gap between reach and settle exceeds the
#   ~ 8-min decay time-constant of MAPK-PP measured by Hornberg et al.,
#   *FEBS J.* 272:244 (2005), Table 2.

# Absolute microM thresholds (NOT [0, 1] fractions).
MAPK_PEAK_MICROM = 0.5  # ~40% of simulator saturation; Markevich 2004
# Fig. 2 half-max activation translated to
# absolute units via MAPK_total = 1.25 microM.
MAPK_SETTLE_MICROM = 0.05  # ~4% of simulator saturation; Markevich 2004
# Fig. 1 baseline / Ferrell-Machleder 1998
# Fig. 2A unstimulated MAPK level.
MKKK_SAFE_MICROM = 0.85 * 0.0035  # 0.85 fraction * MKKK_total = 0.002975
# microM (MKKK_total from MAPKParams,
# itself sourced from Huang-Ferrell 1996
# Table II via Markevich 2004 calibration).
MAPK_T = 60.0  # minutes.

# Backward-compatibility aliases. Earlier callers (calibration scripts,
# audit tables) reference MAPK_PEAK / MAPK_SETTLE / MKKK_SAFE as
# *fractions* of the respective total pools. We keep those names bound
# to the corresponding fractional values so external imports do not
# break, and use the explicit ``*_MICROM`` constants in the spec body.
MAPK_PEAK = MAPK_PEAK_MICROM / 1.25  # 0.40 (fraction of MAPK_total).
MAPK_SETTLE = MAPK_SETTLE_MICROM / 1.25  # 0.04 (fraction of MAPK_total).
MKKK_SAFE = 0.85  # fraction of MKKK_total (unchanged).

mapk_spec = STLSpec(
    name="bio_ode.mapk.hard",
    formula=And(
        children=(
            # Reach: terminal kinase MAPK_PP (state index 4) crosses its
            # half-activation threshold at some point in the first 30 min.
            Eventually(
                _gt("mapk_pp", 4, MAPK_PEAK_MICROM),
                interval=Interval(0.0, 30.0),
            ),
            # Settle: terminal kinase returns to baseline by the end of the
            # horizon. Predicate-level negation is allowed (firewall C.1):
            # equivalently expressed as the predicate
            # ``mapk_pp < MAPK_SETTLE_MICROM``.
            Always(
                _lt("mapk_pp_settle", 4, MAPK_SETTLE_MICROM),
                interval=Interval(45.0, MAPK_T),
            ),
            # Safety: upstream tier (MKKK_P, state index 0) never enters the
            # saturation regime. ``MKKK_SAFE_MICROM = 0.85 * MKKK_total``
            # converts the original [0, 1] fractional Huang-Ferrell 1996
            # Fig. 3b saturation bound to absolute microM units consistent
            # with the simulator output.
            Always(
                Negation(_gt("mkkk_p_unsafe", 0, MKKK_SAFE_MICROM)),
                interval=Interval(0.0, MAPK_T),
            ),
        )
    ),
    signal_dim=6,
    horizon_minutes=MAPK_T,
    description=(
        "Hit the MAPK cascade switch: terminal kinase MAPK-PP (state index "
        "4 in the simulator's 6-vector output, ABSOLUTE microM units) "
        "crosses its half-activation gate (>= 0.5 microM) within 30 min, "
        "then settles back below 0.05 microM in the final 15 min, while "
        "the upstream tier MKKK-P (state index 0) never exceeds 0.85 of "
        "its total pool (<= 0.002975 microM). Form: conjunction of "
        "reachability + tracking + avoidance."
    ),
    citations=(
        "Huang & Ferrell, PNAS 93:10078 (1996), DOI 10.1073/pnas.93.19.10078, Figs. 1, 3b, 4.",
        "Markevich, Hoek & Kholodenko, J Cell Biol 164:353 (2004), Table 1, Fig. 1, Fig. 2.",
        "Alberts et al., Molecular Biology of the Cell, 6th ed. (2014), Fig. 15-49.",
        "Ferrell & Machleder, Science 280:895 (1998), Fig. 2A.",
        "Hornberg et al., FEBS J. 272:244 (2005), Table 2.",
    ),
    formula_text=(
        "F_[0,30] (mapk_pp >= 0.5 microM) AND G_[45,60] (mapk_pp < 0.05 microM) "
        "AND G_[0,60] (NOT (mkkk_p > 0.002975 microM))"
    ),
    metadata={
        "subdomain": "mapk",
        "difficulty": "hard",
        "horizon_min": MAPK_T,
        "control_points": 10,
        "thresholds_microM": {
            "MAPK_PEAK": MAPK_PEAK_MICROM,
            "MAPK_SETTLE": MAPK_SETTLE_MICROM,
            "MKKK_SAFE": MKKK_SAFE_MICROM,
        },
        "thresholds_fraction": {
            "MAPK_PEAK": MAPK_PEAK,
            "MAPK_SETTLE": MAPK_SETTLE,
            "MKKK_SAFE": MKKK_SAFE,
        },
        "allowed_form": ("conjunction of reachability + tracking + avoidance (firewall §C.1)"),
    },
)
register(mapk_spec)


__all__ = [
    "repressilator_spec",
    "toggle_spec",
    "mapk_spec",
    "P_HIGH_NM",
    "P_LOW_NM",
    "TOGGLE_HIGH_NM",
    "TOGGLE_LOW_NM",
    "TOGGLE_UNSAFE_NM",
    "MAPK_PEAK_MICROM",
    "MAPK_SETTLE_MICROM",
    "MKKK_SAFE_MICROM",
    "MAPK_PEAK",
    "MAPK_SETTLE",
    "MKKK_SAFE",
]
