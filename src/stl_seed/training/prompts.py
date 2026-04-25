"""Jinja2 system-prompt templates for stl-seed task families.

One template per task family in the canonical 3×3×2 sweep:

* ``repressilator`` — Elowitz-Leibler 3-gene oscillator (Elowitz & Leibler
  2000, DOI:10.1038/35002125). Action = transcription-rate scaling factors.
* ``toggle`` — Gardner-Cantor-Collins genetic toggle switch (Gardner et al.
  2000, DOI:10.1038/35002131). Action = inducer concentrations.
* ``mapk`` — Huang-Ferrell 1996 MAPK cascade (Huang & Ferrell 1996,
  DOI:10.1073/pnas.93.19.10078). Action = upstream stimulus magnitudes.
* ``glucose_insulin`` — Bergman 1979 minimal model + Dalla Man 2007
  (Bergman et al. 1979, DOI:10.1152/ajpendo.1979.236.6.E667). Action =
  insulin infusion rates.

Each template explains the task in plain English, lists the action-space
schema (dimensions and physical units), and locks the I/O serialization
format the tokenizer downstream will produce. The format is shared across
all families:

    <state>v1,v2,...,vn</state><action>u1,u2,...,um</action>

This matches SERA's chat-template convention (paper/REDACTED.md §B.5)
where the assistant turn carries a per-step structured response. Keeping
the format fixed across families means the model can transfer
"observation → action" tokenization patterns between tasks even if the
physics changes.

The templates are rendered by ``stl_seed.training.tokenize`` at dataset-
construction time — never at training time — so any rendering error
surfaces before SFT begins.
"""

from __future__ import annotations

from typing import Final

import jinja2

# ---------------------------------------------------------------------------
# Template environment.
# ---------------------------------------------------------------------------


_ENV: Final[jinja2.Environment] = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    autoescape=False,  # plain text prompts; no HTML escaping
    keep_trailing_newline=False,
    undefined=jinja2.StrictUndefined,  # fail loudly on missing template vars
    trim_blocks=True,
    lstrip_blocks=True,
)


# ---------------------------------------------------------------------------
# Per-family templates.
# ---------------------------------------------------------------------------


REPRESSILATOR_SYSTEM_PROMPT: Final[str] = """\
You are a controller for a 3-gene synthetic genetic oscillator (the
Elowitz-Leibler repressilator). Three repressor proteins (LacI, TetR, cI)
mutually inhibit each other in a cyclic loop. Your job is to emit a
sequence of transcription-rate scaling actions that drive the system to
satisfy the temporal-logic specification given below.

State (3-D, in nM):
  - protein concentration of LacI, TetR, cI

Action (3-D, dimensionless multiplicative scaling, clipped to [0.1, 10.0]):
  - transcription-rate multiplier for each of the three genes

Specification: {{ spec_text }}

Horizon: {{ horizon }} control steps over {{ duration_minutes }} minutes
of simulated time.

At each step you observe the current state and emit one action. Use the
exact format:

  <state>v1,v2,v3</state><action>u1,u2,u3</action>

Numbers in scientific notation with 4 significant figures. Do not emit
any other text.
"""


TOGGLE_SYSTEM_PROMPT: Final[str] = """\
You are a controller for the Gardner-Cantor-Collins genetic toggle
switch. Two repressors (LacI and TetR) mutually inhibit each other; in
the absence of inducer, the system locks into one of two stable states.
You may add inducers (IPTG to relieve LacI; aTc to relieve TetR) to
push the switch between states.

State (2-D, in nM):
  - protein concentration of LacI, TetR

Action (2-D, in micromolar inducer concentration, clipped to [0.0, 100.0]):
  - IPTG concentration, aTc concentration

Specification: {{ spec_text }}

Horizon: {{ horizon }} control steps over {{ duration_minutes }} minutes
of simulated time.

At each step you observe the current state and emit one action. Use the
exact format:

  <state>v1,v2</state><action>u1,u2</action>

Numbers in scientific notation with 4 significant figures. Do not emit
any other text.
"""


MAPK_SYSTEM_PROMPT: Final[str] = """\
You are a controller for the Huang-Ferrell MAPK signaling cascade. A
three-tier kinase cascade (MAPKKK → MAPKK → MAPK) amplifies an upstream
stimulus into a downstream phosphorylation response. Your job is to
modulate the stimulus signal to drive the doubly-phosphorylated MAPK
output to satisfy the temporal-logic specification.

State (5-D, normalized phosphorylation fraction in [0, 1]):
  - MAPKKK-P, MAPKK-P, MAPKK-PP, MAPK-P, MAPK-PP

Action (1-D, dimensionless stimulus magnitude, clipped to [0.0, 10.0]):
  - input signal strength to MAPKKK

Specification: {{ spec_text }}

Horizon: {{ horizon }} control steps over {{ duration_minutes }} minutes
of simulated time.

At each step you observe the current state and emit one action. Use the
exact format:

  <state>v1,v2,v3,v4,v5</state><action>u1</action>

Numbers in scientific notation with 4 significant figures. Do not emit
any other text.
"""


GLUCOSE_INSULIN_SYSTEM_PROMPT: Final[str] = """\
You are a closed-loop insulin controller for the Bergman minimal model
of glucose-insulin dynamics, augmented with the Dalla Man 2007 meal
absorption model. You observe plasma glucose and remote insulin and
emit an insulin infusion rate to keep the patient inside the target
glycemic envelope specified below.

State (3-D):
  - plasma glucose G (mg/dL)
  - remote insulin action X (1/min)
  - plasma insulin I (mU/L)

Action (1-D, in mU/min, clipped to [0.0, 50.0]):
  - exogenous insulin infusion rate

Specification: {{ spec_text }}

Horizon: {{ horizon }} control steps over {{ duration_minutes }} minutes
of simulated time.

At each step you observe the current state and emit one action. Use the
exact format:

  <state>v1,v2,v3</state><action>u1</action>

Numbers in scientific notation with 4 significant figures. Do not emit
any other text.
"""


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


_TEMPLATES: Final[dict[str, str]] = {
    "repressilator": REPRESSILATOR_SYSTEM_PROMPT,
    "toggle": TOGGLE_SYSTEM_PROMPT,
    "mapk": MAPK_SYSTEM_PROMPT,
    "glucose_insulin": GLUCOSE_INSULIN_SYSTEM_PROMPT,
}


def render_system_prompt(
    task: str,
    spec_text: str,
    horizon: int,
    duration_minutes: float,
) -> str:
    """Render the system prompt for a given task family.

    Parameters
    ----------
    task:
        Task family name; one of ``repressilator``, ``toggle``, ``mapk``,
        ``glucose_insulin``.
    spec_text:
        Human-readable STL formula text (``STLSpec.formula_text`` from
        ``stl_seed.specs``).
    horizon:
        Number of control steps ``H`` (matches ``Trajectory.actions.shape[0]``).
    duration_minutes:
        Total simulated horizon in minutes (``STLSpec.horizon_minutes``).

    Raises
    ------
    KeyError:
        If ``task`` is not registered.
    jinja2.UndefinedError:
        If a required template variable is missing (StrictUndefined).
    """
    if task not in _TEMPLATES:
        raise KeyError(
            f"Unknown task family {task!r}; registered: {sorted(_TEMPLATES)}"
        )
    tmpl = _ENV.from_string(_TEMPLATES[task])
    return tmpl.render(
        spec_text=spec_text,
        horizon=horizon,
        duration_minutes=duration_minutes,
    )


def list_tasks() -> list[str]:
    """Return the list of registered task family names (sorted)."""
    return sorted(_TEMPLATES)


__all__ = [
    "REPRESSILATOR_SYSTEM_PROMPT",
    "TOGGLE_SYSTEM_PROMPT",
    "MAPK_SYSTEM_PROMPT",
    "GLUCOSE_INSULIN_SYSTEM_PROMPT",
    "render_system_prompt",
    "list_tasks",
]
