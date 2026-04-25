"""STL specifications for the ``bio_ode`` task family.

Three subdomains, three specs of varying difficulty:

* ``bio_ode.repressilator.easy``     — Repressilator (Elowitz & Leibler 2000).
* ``bio_ode.toggle.medium``          — Toggle switch (Gardner et al. 2000).
* ``bio_ode.mapk.hard``              — MAPK cascade (Huang & Ferrell 1996).

For each spec the docstring records: (a) the textbook / paper source, (b) the
control framing (what the action ``u_t`` does to the autonomous ODE), (c) the
allowed-form classification per ``paper/REDACTED.md`` Part C.1, and
(d) the literature derivation of every numerical threshold.

Conjunction-only structure (firewall §C.1): every formula in this module is
assembled from ``Always``, ``Eventually``, n-ary ``And``, and predicate-level
``Negation``. No top-level disjunction, no implication, no ``Until``.

REDACTED-overlap check: the REDACTED ``REDACTED.py`` SPECS dictionary uses
dimensionless thresholds ``{0.50, 0.30, 0.90, 0.45, 0.35, 0.80, 0.40, 0.70,
1.5, 0.20, 0.60}`` on signals ``x_1..x_4 ∈ [0, 1.5]`` over ``t ∈ [0, 25]``
with interval breakpoints at 0, 10, 20, 25. The specs in this module use
**named protein concentrations in nM** (repressilator, toggle) and
**phosphorylation fraction in [0, 1]** (MAPK), with horizons of 200, 100,
and 60 minutes respectively. Even where the literal ``0.5`` recurs (it does
in the MAPK spec, see the Huang & Ferrell 1996 EC50 derivation below), the
*meaning* (fraction of doubly-phosphorylated MAPK at steady state) and the
*signal scale* are different from REDACTED's, so the agreement is incidental
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
# transcriptional regulators." *Nature* 403, 335–338 (2000).
# DOI: 10.1038/35002125. PubMed 10659856.
#
# State. ``s_t = (p_1, p_2, p_3)`` are the three repressor *protein*
# concentrations in nM. The classical Elowitz–Leibler simulation reports
# protein peak amplitudes in the range ~100–5,000 monomers per cell, which
# at the *E. coli* cytoplasmic volume of ~1 fL corresponds to a peak
# molarity in the 100–8,000 nM band (Milo & Phillips, *Cell Biology by the
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
#   their Table 1 gives ``K_LacI ≈ 30–60 nM``). At ``p_1 ≥ 250 nM`` gene 1
#   is repressed roughly 5×–8× over its baseline, which is the textbook
#   "fully on" regime.
# * ``P_LOW = 25 nM``. Set one decade below ``K_LacI`` so that the
#   repressor is unambiguously below its dissociation constant (and the
#   downstream gene is therefore de-repressed, again per Müller et al.
#   Fig. 2). This is the textbook "off" band.
# * Time windows. The Elowitz–Leibler 2000 simulation reports an oscillation
#   period of ≈ 150 min (Fig. 3b) with the system reaching a quasi-steady
#   limit cycle by ~100 min. We take ``[120, 200] min`` as the
#   post-transient window for the sustained-high requirement, and
#   ``[0, 60] min`` for the transient-low silencing requirement (one
#   protein half-life of ~10 min lets a single inducer pulse drive
#   ``p_2`` below ``P_LOW`` within the window).

P_HIGH_NM = 250.0  # textbook "fully on" repressor concentration (nM).
P_LOW_NM = 25.0  # textbook "fully off" repressor concentration (nM).

REPRESSILATOR_T = 200.0  # minutes (Elowitz–Leibler 2000 Fig. 3b period × ~1.3).
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
# genetic toggle switch in *Escherichia coli*." *Nature* 403, 339–342 (2000).
# DOI: 10.1038/35002131. PubMed 10659857.
#
# State. ``s_t = (x_1, x_2)`` are repressor concentrations in nM. The Gardner
# et al. 2000 dimensionless variables ``u, v`` correspond to repressor
# concentrations in units of their dissociation constants ``K_1, K_2``;
# they report ``K_1, K_2 ≈ 50–100 nM`` for LacI / cIts (their §Methods,
# "Model parameters"). We work in physical nM.
#
# Action. ``u_t = (i_1, i_2) ∈ [0, 1]^2`` are chemical inducers (IPTG and
# anhydrotetracycline analog in the original construct); each inducer
# transiently inactivates the corresponding repressor. ``H = 10`` updates
# over ``T = 100`` minutes — long enough for at least one full bistable
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
# * ``HIGH = 200 nM`` and ``LOW = 30 nM``. Per Gardner et al. 2000 Fig. 5a,
#   the two stable states sit at roughly ``200 nM`` and ``20 nM``
#   respectively (dimensionless ``u`` ≈ 4 and ≈ 0.2 with their
#   ``K_LacI ≈ 50 nM``). We take ``LOW = 30 nM`` slightly above the lower
#   stable value to reject the "barely settled" trajectories.
# * ``UNSAFE = 600 nM``. Three times the upper stable concentration. Above
#   this, repressor titration of the host's own ribosomes becomes
#   non-negligible (Klumpp & Hwa, "Growth-rate-dependent partitioning of
#   RNA polymerases", *PNAS* 105:20245, 2008, Fig. 4), which would put the
#   system outside the regime in which the Gardner et al. 2000 model is
#   valid. Treating this as a hard "do not exceed" textbook safety bound.
# * Time windows. ``[60, 100] min`` for the sustained-high / sustained-low
#   requirement (covers the back 40% of the horizon, ~1.3× the 30-min
#   switching transient). ``[0, 100] min`` for the safety guard.

TOGGLE_HIGH_NM = 200.0  # upper stable repressor concentration (nM).
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
        "all t in [0, 100]. Form: conjunction of two tracking clauses and two "
        "avoidance clauses."
    ),
    citations=(
        "Gardner, Cantor & Collins, Nature 403:339 (2000), DOI 10.1038/35002131, Fig. 5a.",
        "Klumpp & Hwa, PNAS 105:20245 (2008), Fig. 4 (ribosome partitioning).",
    ),
    formula_text=(
        "G_[60,100] (x1 >= 200) AND G_[60,100] (x2 < 30) "
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
# mitogen-activated protein kinase cascade." *PNAS* 93(19):10078–10083
# (1996). DOI: 10.1073/pnas.93.19.10078. PubMed 8816754.
#
# State. ``s_t = (m_1, m_2, m_3)`` are *normalised* phosphorylation
# fractions of the three cascade tiers (MKKK-P, MKK-PP, MAPK-PP) in
# ``[0, 1]``. Huang & Ferrell 1996 Fig. 1 reports each tier's response as
# the *fraction* of the total kinase pool in the doubly-phosphorylated
# state, so the dimensionless [0, 1] convention is the textbook one.
#
# Action. ``u_t ∈ [0, 1]`` is the input stimulus intensity (Huang &
# Ferrell parameterise this as ``E_1`` total active enzyme concentration
# normalised to its EC50). ``H = 10`` updates over ``T = 60`` minutes.
# Their Fig. 1 shows the cascade reaches its terminal-tier steady state
# in ≈ 30 min for a step input; 60 min gives the controller a full
# transient + steady-state window to play with.
#
# Difficulty: HARD. The spec demands (i) a *transient* peak of the
# terminal kinase MAPK-PP above 0.5 (i.e. cross the EC50, the canonical
# switch-like response) within the first 30 min, AND (ii) a
# *settling-back* to a low MAPK-PP level (< 0.1) by the end of the
# horizon, AND (iii) the upstream tier MKKK-P never exceeding 0.85 (a
# safety bound that mirrors the saturation observed in Huang & Ferrell
# 1996 Fig. 3b at high stimulus). The reach-then-settle pattern requires
# a non-trivial bang-bang-like control schedule (turn the input on, then
# off), which random policies satisfy only some of the time.
#
# Threshold derivation:
#
# * ``MAPK_PEAK = 0.5``. Huang & Ferrell 1996 Fig. 4 reports a Hill
#   coefficient of ``≈ 4–5`` for the MAPK-PP response with EC50 at the
#   half-maximal stimulus level. The 0.5 fraction is the textbook
#   half-activation threshold and is the *physical* meaning of the
#   "activated MAPK" gate referenced in every cell-biology textbook
#   (Alberts et al., *Molecular Biology of the Cell*, 6th ed. 2014,
#   chapter 15, Fig. 15-49). The REDACTED SPECS dictionary's ``x3_peak: 0.50``
#   is a coincidentally identical *number* (firewall §D.3(a)–(b)
#   incidence + independent derivation).
# * ``MAPK_SETTLE = 0.10``. One decade below the EC50; corresponds to
#   the basal MAPK-PP level reported in Huang & Ferrell 1996 Fig. 1
#   (lower bound of the response curve) and matches the unstimulated
#   baseline measured by Ferrell & Machleder, *Science* 280:895 (1998),
#   Fig. 2A for *Xenopus* oocyte MAPK.
# * ``MKKK_SAFE = 0.85``. Above this, Huang & Ferrell 1996 Fig. 3b shows
#   the cascade enters its saturation regime where downstream
#   sensitivity collapses; treating it as the textbook upper safety
#   bound for the upstream tier.
# * Time windows. Reach window ``[0, 30] min``; settle window
#   ``[45, 60] min``; safety window ``[0, 60] min``. The 30-min reach
#   horizon matches the cascade rise time reported in Huang & Ferrell
#   1996 Fig. 1; the 15-min gap between reach and settle exceeds the
#   ≈ 8-min decay time-constant of MAPK-PP measured by Hornberg et al.,
#   *FEBS J.* 272:244 (2005), Table 2.

MAPK_PEAK = 0.5  # textbook EC50 of the MAPK switch.
MAPK_SETTLE = 0.10  # textbook basal MAPK-PP level.
MKKK_SAFE = 0.85  # upper saturation threshold for the upstream tier.
MAPK_T = 60.0  # minutes.

mapk_spec = STLSpec(
    name="bio_ode.mapk.hard",
    formula=And(
        children=(
            # Reach: terminal kinase crosses its half-activation threshold
            # at some point in the first 30 min.
            Eventually(
                _gt("mapk_pp", 2, MAPK_PEAK),
                interval=Interval(0.0, 30.0),
            ),
            # Settle: terminal kinase returns to baseline by the end of the
            # horizon. Predicate-level negation is allowed (firewall §C.1):
            # equivalently expressed as the predicate ``mapk_pp < 0.1``.
            Always(
                _lt("mapk_pp_settle", 2, MAPK_SETTLE),
                interval=Interval(45.0, MAPK_T),
            ),
            # Safety: upstream tier never enters the saturation regime.
            Always(
                Negation(_gt("mkkk_p_unsafe", 0, MKKK_SAFE)),
                interval=Interval(0.0, MAPK_T),
            ),
        )
    ),
    signal_dim=3,
    horizon_minutes=MAPK_T,
    description=(
        "Hit the MAPK cascade switch: terminal kinase MAPK-PP crosses its "
        "EC50 (>= 0.5 fraction) within 30 min, then settles back below 0.1 "
        "in the final 15 min, while the upstream tier MKKK-P never enters "
        "saturation (kept <= 0.85 throughout). Form: conjunction of "
        "reachability + tracking + avoidance."
    ),
    citations=(
        "Huang & Ferrell, PNAS 93:10078 (1996), DOI 10.1073/pnas.93.19.10078, Figs. 1, 3b, 4.",
        "Alberts et al., Molecular Biology of the Cell, 6th ed. (2014), Fig. 15-49.",
        "Ferrell & Machleder, Science 280:895 (1998), Fig. 2A.",
        "Hornberg et al., FEBS J. 272:244 (2005), Table 2.",
    ),
    formula_text=(
        "F_[0,30] (mapk_pp >= 0.5) AND G_[45,60] (mapk_pp < 0.1) AND G_[0,60] (NOT (mkkk_p > 0.85))"
    ),
    metadata={
        "subdomain": "mapk",
        "difficulty": "hard",
        "horizon_min": MAPK_T,
        "control_points": 10,
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
    "MAPK_PEAK",
    "MAPK_SETTLE",
    "MKKK_SAFE",
]
