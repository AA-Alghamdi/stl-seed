"""STL specifications for the ``glucose_insulin`` task family.

Three specs of varying difficulty over the same Bergman 1979 + Dalla Man 2007
state ``s_t = (G, X, I)``:

* ``glucose_insulin.tir.easy``       — Time-in-Range (ADA 2024 standards).
* ``glucose_insulin.no_hypo.medium`` — TIR + strict hypoglycaemia avoidance.
* ``glucose_insulin.dawn.hard``      — Reach-then-track around a postprandial
  challenge with a transient insulin-bolus reachability requirement.

State conventions:

* ``G`` — plasma glucose, mg/dL. (Bergman et al. 1979 use mg/dL throughout
  their Eq. 1, and the ADA 2024 Standards of Care Time-in-Range targets are
  reported in mg/dL.)
* ``X`` — remote-compartment insulin action, normalised arbitrary units
  per Bergman et al. 1979 Eq. 1.
* ``I`` — plasma insulin, µU/mL. Bergman et al. 1979 Table 1 reports
  basal ``I_b ≈ 7–15 µU/mL`` for healthy adults.

Action: ``u_t ∈ [0, 5] U/h`` insulin infusion rate (a clinically realistic
upper bound for closed-loop insulin pumps; see Garg et al., *Diabetes Tech.
Ther.* 21:155 (2019), Table 2). ``H = 12`` updates over ``T = 120`` minutes
(one update every 10 min, matching the canonical CGM sampling cadence).

Conjunction-only structure (firewall §C.1): every formula in this module
uses only ``Always``, ``Eventually``, n-ary ``And``, and predicate-level
``Negation``. No top-level disjunction, no implication, no ``Until``.

REDACTED-overlap check: the REDACTED ``REDACTED.py`` SPECS dictionary uses
dimensionless thresholds in [0, 1.5] over t in [0, 25] dimensionless time
units on dimensionless gene-expression signals. The thresholds in this
module are *clinical glucose values in mg/dL* (54, 70, 140, 180, 250) and
*plasma insulin in µU/mL* (40, 100), with horizons of 120 minutes. There is
zero numerical overlap with the REDACTED literal set (verified by inspection of
``REDACTED.py`` SPECS dict and ``REDACTED.py:447–451``).
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
# Clinical thresholds (single source of truth across the three specs).
# ---------------------------------------------------------------------------
#
# All values are mg/dL plasma glucose unless otherwise stated. Sourced from:
#
# * American Diabetes Association, "6. Glycemic Targets: Standards of Care
#   in Diabetes — 2024." *Diabetes Care* 47(Suppl. 1):S111–S125 (2024).
#   DOI: 10.2337/dc24-S006.
# * Battelino, T. *et al.* "Clinical targets for continuous glucose
#   monitoring data interpretation: recommendations from the international
#   consensus on time in range." *Diabetes Care* 42:1593 (2019).
#   DOI: 10.2337/dci19-0028.
# * Whipple, A. O. "The surgical therapy of hyperinsulinism." *J. Int.
#   Chir.* 3:237 (1938) — defines clinical hypoglycaemia (the "Whipple
#   triad" lower bound).
# * Bergman, R. N., Ider, Y. Z., Bowden, C. R. & Cobelli, C. "Quantitative
#   estimation of insulin sensitivity." *Am. J. Physiol.* 236:E667 (1979).
#   PubMed 443421.

G_TIR_LOW = 70.0  # mg/dL — lower bound of ADA 2024 target range.
G_TIR_HIGH = 180.0  # mg/dL — upper bound of ADA 2024 target range.
G_HYPO_SEVERE = 54.0  # mg/dL — Battelino 2019 "Level 2" severe hypo.
G_HYPER_LEVEL2 = 250.0  # mg/dL — Battelino 2019 "Level 2" hyperglycaemia.
G_POSTPRANDIAL_PEAK = 140.0  # mg/dL — ADA 2024 2h-postprandial target.

I_BOLUS_MIN = 40.0  # µU/mL — minimum effective bolus peak (see derivation).
I_HYPER_HIGH = 100.0  # µU/mL — upper safety bound on plasma insulin.

GI_T = 120.0  # minutes — full simulation horizon.


# ---------------------------------------------------------------------------
# Spec 1 (EASY): Time-in-Range tracking.
# ---------------------------------------------------------------------------
#
# Textbook source. ADA 2024 Standards of Care Recommendation 6.5b: "An A1C
# goal of <7% (53 mmol/mol) is appropriate for many nonpregnant adults with
# diabetes" with TIR target of 70–180 mg/dL >70% of the time. We weaken
# the >70% statistical TIR target to a *deterministic* "always within
# range over [30, 120] min" because (a) we are evaluating individual
# trajectories rather than aggregates, and (b) the first 30 min are the
# absorptive phase of a meal in the Dalla Man 2007 model where excursions
# above 180 mg/dL are physiologically expected (Dalla Man et al. 2007 Fig.
# 4 shows postprandial peak at ~30 min).
#
# Difficulty: EASY. A constant infusion rate near the basal needs of the
# Dalla Man healthy-subject template (~1–2 U/h) keeps glucose in range
# most of the time; random infusion sequences satisfy the spec a
# meaningful fraction of the time.

tir_spec = STLSpec(
    name="glucose_insulin.tir.easy",
    formula=And(
        children=(
            Always(
                _gt("G_above_70", 0, G_TIR_LOW),
                interval=Interval(30.0, GI_T),
            ),
            Always(
                _lt("G_below_180", 0, G_TIR_HIGH),
                interval=Interval(30.0, GI_T),
            ),
        )
    ),
    signal_dim=3,
    horizon_minutes=GI_T,
    description=(
        "Hold plasma glucose within the ADA 2024 Time-in-Range band "
        "[70, 180] mg/dL throughout the post-absorptive window [30, 120] min. "
        "Form: conjunction of two tracking clauses (lower and upper bound)."
    ),
    citations=(
        "ADA 2024 Standards of Care, Recommendation 6.5b, "
        "Diabetes Care 47(Suppl. 1):S111 (2024), DOI 10.2337/dc24-S006.",
        "Battelino et al., Diabetes Care 42:1593 (2019), Table 1, "
        "DOI 10.2337/dci19-0028.",
        "Dalla Man, Rizza & Cobelli, IEEE TBME 54:1740 (2007), Fig. 4.",
    ),
    formula_text=(
        "G_[30,120] (G >= 70) AND G_[30,120] (G < 180)"
    ),
    metadata={
        "difficulty": "easy",
        "horizon_min": GI_T,
        "control_points": 12,
        "thresholds_mgdl": {"LOW": G_TIR_LOW, "HIGH": G_TIR_HIGH},
        "allowed_form": "conjunction of two tracking clauses (firewall §C.1)",
    },
)
register(tir_spec)


# ---------------------------------------------------------------------------
# Spec 2 (MEDIUM): TIR + strict hypoglycaemia avoidance.
# ---------------------------------------------------------------------------
#
# Textbook source. The hypoglycaemia avoidance clause encodes Whipple's
# triad lower bound (Whipple 1938) at the modern Battelino 2019 "Level 2"
# severe-hypo threshold of 54 mg/dL. The hyperglycaemia avoidance clause
# uses Battelino 2019 "Level 2" hyperglycaemia threshold of 250 mg/dL.
# These are textbook clinical bright lines that any closed-loop insulin
# controller must respect, independent of the soft TIR objective.
#
# Difficulty: MEDIUM. The spec adds two hard avoidance bounds on top of
# the easy TIR spec. Random over-dosing trajectories that hit the 54 mg/dL
# severe-hypoglycaemia floor — a clinically catastrophic outcome —
# automatically violate the spec, which sharply reduces the random-policy
# success rate relative to the easy spec.

no_hypo_spec = STLSpec(
    name="glucose_insulin.no_hypo.medium",
    formula=And(
        children=(
            # TIR window (same as the easy spec).
            Always(
                _gt("G_above_70", 0, G_TIR_LOW),
                interval=Interval(30.0, GI_T),
            ),
            Always(
                _lt("G_below_180", 0, G_TIR_HIGH),
                interval=Interval(30.0, GI_T),
            ),
            # Severe-hypo avoidance over the full horizon (Whipple 1938 +
            # Battelino 2019 Level 2). Encoded with predicate-level negation
            # of "G is below 54", which is exactly the textbook reading of
            # the avoidance.
            Always(
                Negation(_lt("G_severe_hypo", 0, G_HYPO_SEVERE)),
                interval=Interval(0.0, GI_T),
            ),
            # Severe-hyper avoidance (Battelino 2019 Level 2).
            Always(
                Negation(_gt("G_severe_hyper", 0, G_HYPER_LEVEL2)),
                interval=Interval(0.0, GI_T),
            ),
        )
    ),
    signal_dim=3,
    horizon_minutes=GI_T,
    description=(
        "ADA 2024 Time-in-Range [70, 180] mg/dL on [30, 120] min, with "
        "absolute prohibition on Level-2 severe hypoglycaemia (G < 54 mg/dL) "
        "and Level-2 severe hyperglycaemia (G > 250 mg/dL) at any point in "
        "[0, 120] min. Form: conjunction of two tracking clauses and two "
        "avoidance clauses."
    ),
    citations=(
        "ADA 2024 Standards of Care, Recommendation 6.5b, "
        "Diabetes Care 47(Suppl. 1):S111 (2024).",
        "Battelino et al., Diabetes Care 42:1593 (2019), Table 1.",
        "Whipple, J. Int. Chir. 3:237 (1938).",
    ),
    formula_text=(
        "G_[30,120] (G >= 70) AND G_[30,120] (G < 180) "
        "AND G_[0,120] (NOT (G < 54)) AND G_[0,120] (NOT (G > 250))"
    ),
    metadata={
        "difficulty": "medium",
        "horizon_min": GI_T,
        "control_points": 12,
        "thresholds_mgdl": {
            "TIR_LOW": G_TIR_LOW,
            "TIR_HIGH": G_TIR_HIGH,
            "HYPO_SEVERE": G_HYPO_SEVERE,
            "HYPER_LEVEL2": G_HYPER_LEVEL2,
        },
        "allowed_form": (
            "conjunction of tracking + avoidance "
            "(firewall §C.1)"
        ),
    },
)
register(no_hypo_spec)


# ---------------------------------------------------------------------------
# Spec 3 (HARD): Postprandial reach-then-track ("dawn-phenomenon" challenge).
# ---------------------------------------------------------------------------
#
# Textbook source. The postprandial 2-hour glucose target of 140 mg/dL is
# the ADA 2024 Recommendation 6.5b postprandial bound; combined with the
# Dalla Man 2007 meal-model reach pattern (Fig. 4 shows insulin peaks at
# ~30 min after meal onset) this gives a concrete reach-then-track spec.
# The minimum effective insulin bolus peak ``I_BOLUS_MIN = 40 µU/mL`` is
# derived from Polonsky et al., "Twenty-four-hour profiles and pulsatile
# patterns of insulin secretion in normal and obese subjects." *J. Clin.
# Invest.* 81:442 (1988), Fig. 1: peak postprandial insulin in healthy
# subjects ≈ 40–80 µU/mL. We require the controller to produce a bolus
# whose plasma-insulin peak crosses 40 µU/mL within the first hour, then
# settle back below the 100 µU/mL upper safety bound (Cryer, "Hypoglycemia
# in Diabetes: Pathophysiology, Prevalence, and Prevention," ADA 2016
# monograph, §3, defines "iatrogenic hyperinsulinaemia" above this level).
#
# Difficulty: HARD. The spec requires (i) hitting a non-trivial insulin
# peak, (ii) returning glucose to the 2-hour postprandial target band
# [70, 140] mg/dL by t = 120 min, and (iii) never breaching the
# severe-hypo or severe-hyper bounds. A random infusion sequence with no
# coordination between bolus and meal absorption rarely satisfies all
# three simultaneously, putting the random-policy success rate at the low
# end of the calibration band.

dawn_spec = STLSpec(
    name="glucose_insulin.dawn.hard",
    formula=And(
        children=(
            # Reachability: deliver a clinically meaningful insulin bolus
            # within the first hour (peak plasma insulin >= 40 µU/mL at
            # some point in [10, 60] min). Channel index 2 = I.
            Eventually(
                _gt("I_bolus", 2, I_BOLUS_MIN),
                interval=Interval(10.0, 60.0),
            ),
            # Tracking: 2-hour postprandial glucose returns to [70, 140].
            Always(
                _gt("G_post_above_70", 0, G_TIR_LOW),
                interval=Interval(90.0, GI_T),
            ),
            Always(
                _lt("G_post_below_140", 0, G_POSTPRANDIAL_PEAK),
                interval=Interval(90.0, GI_T),
            ),
            # Safety: severe-hypo never, severe-hyper never, plasma insulin
            # never above the iatrogenic-hyperinsulinaemia line.
            Always(
                Negation(_lt("G_no_severe_hypo", 0, G_HYPO_SEVERE)),
                interval=Interval(0.0, GI_T),
            ),
            Always(
                Negation(_gt("G_no_severe_hyper", 0, G_HYPER_LEVEL2)),
                interval=Interval(0.0, GI_T),
            ),
            Always(
                _lt("I_safe", 2, I_HYPER_HIGH),
                interval=Interval(0.0, GI_T),
            ),
        )
    ),
    signal_dim=3,
    horizon_minutes=GI_T,
    description=(
        "Postprandial reach-then-track: deliver an effective insulin bolus "
        "(plasma insulin peak >= 40 µU/mL within the first hour), bring "
        "glucose back to the ADA 2-hour postprandial target band "
        "[70, 140] mg/dL on [90, 120] min, and never breach the severe-hypo "
        "(54 mg/dL), severe-hyper (250 mg/dL), or iatrogenic-hyperinsulin "
        "(100 µU/mL) safety bounds. Form: conjunction of reachability + two "
        "tracking + three avoidance clauses."
    ),
    citations=(
        "ADA 2024 Standards of Care, Recommendation 6.5b "
        "(2-h postprandial 140 mg/dL), Diabetes Care 47(Suppl. 1):S111 (2024).",
        "Battelino et al., Diabetes Care 42:1593 (2019), Table 1.",
        "Whipple, J. Int. Chir. 3:237 (1938).",
        "Polonsky et al., J. Clin. Invest. 81:442 (1988), Fig. 1.",
        "Cryer, ADA Monograph: Hypoglycemia in Diabetes (2016), §3.",
        "Dalla Man, Rizza & Cobelli, IEEE TBME 54:1740 (2007), Fig. 4.",
    ),
    formula_text=(
        "F_[10,60] (I >= 40) AND G_[90,120] (G >= 70) "
        "AND G_[90,120] (G < 140) AND G_[0,120] (NOT (G < 54)) "
        "AND G_[0,120] (NOT (G > 250)) AND G_[0,120] (I < 100)"
    ),
    metadata={
        "difficulty": "hard",
        "horizon_min": GI_T,
        "control_points": 12,
        "thresholds_mgdl": {
            "TIR_LOW": G_TIR_LOW,
            "POSTPRANDIAL_PEAK": G_POSTPRANDIAL_PEAK,
            "HYPO_SEVERE": G_HYPO_SEVERE,
            "HYPER_LEVEL2": G_HYPER_LEVEL2,
        },
        "thresholds_uU_mL": {
            "I_BOLUS_MIN": I_BOLUS_MIN,
            "I_HYPER_HIGH": I_HYPER_HIGH,
        },
        "allowed_form": (
            "conjunction of reachability + tracking + avoidance "
            "(firewall §C.1)"
        ),
    },
)
register(dawn_spec)


__all__ = [
    "tir_spec",
    "no_hypo_spec",
    "dawn_spec",
    "G_TIR_LOW",
    "G_TIR_HIGH",
    "G_HYPO_SEVERE",
    "G_HYPER_LEVEL2",
    "G_POSTPRANDIAL_PEAK",
    "I_BOLUS_MIN",
    "I_HYPER_HIGH",
    "GI_T",
]
