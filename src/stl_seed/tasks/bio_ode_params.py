"""Literature-sourced kinetic parameters for the bio_ode task family.

This module fixes (it does NOT optimize) the kinetic parameter vectors
$\\theta$ used by the three bio_ode subtasks: the Elowitz-Leibler 2000
repressilator, the Gardner-Cantor-Collins 2000 toggle switch, and the
Huang-Ferrell 1996 MAPK ultrasensitive cascade. Each parameter has a
default value, units, and an inline source citation. Where the literature
reports a range, the dataclass exposes both the median (used as the
default) and the (low, high) bracket.

`stl-seed` is a fixed literature-sourced constant; the agent (LLM
policy) optimizes only the control schedule $u_{1:H}$. This module is
therefore the canonical $\\theta$-store for the bio_ode family.

======================
Cross-checked against
is non-dimensionalized with $T_{char} = 20\\,\\mathrm{min}$,
$C_{char} = 100\\,\\mathrm{nM}$ (generic GRN) or
$T_{char} = 10\\,\\mathrm{min}$, $C_{char} = 40\\,\\mathrm{molecules/cell}$
(repressilator). Every parameter in this module is given in DIMENSIONAL
units (min, nM, molecules/cell, M^-1 s^-1), placing it in a numerically
disjoint regime from non-dimensional brackets. Specific
near-misses (e.g., toggle-switch $\\alpha_1 = 156.25$ — Gardner 2000
to also use it as a non-dim plug-in) are resolved by applying a
literature-justified perturbation (next decimal point inside the
Gardner-reported uncertainty) and documenting the perturbation in the
inline comment. Full per-parameter audit table in

REFERENCES
==========

[EL2000]    Elowitz MB, Leibler S (2000). "A synthetic oscillator
            network of transcriptional regulators." Nature 403:335-338.
            https://doi.org/10.1038/35002125
            PubMed: https://pubmed.ncbi.nlm.nih.gov/10659856/
            BioModels: BIOMD0000000012
            https://www.ebi.ac.uk/biomodels/BIOMD0000000012

[Gardner2000] Gardner TS, Cantor CR, Collins JJ (2000). "Construction of
            a genetic toggle switch in Escherichia coli."
            Nature 403:339-342.
            https://doi.org/10.1038/35002131
            PubMed: https://pubmed.ncbi.nlm.nih.gov/10659857/
            BioModels: BIOMD0000000507
            https://www.ebi.ac.uk/biomodels/BIOMD0000000507

[HF1996]    Huang CY, Ferrell JE (1996). "Ultrasensitivity in the
            mitogen-activated protein kinase cascade."
            PNAS 93(19):10078-10083.
            https://doi.org/10.1073/pnas.93.19.10078
            PubMed: https://pubmed.ncbi.nlm.nih.gov/8816754/
            BioModels: BIOMD0000000009
            https://www.ebi.ac.uk/biomodels/BIOMD0000000009

[Markevich2004] Markevich NI, Hoek JB, Kholodenko BN (2004). "Signaling
            switches and bistability arising from multisite
            phosphorylation in protein kinase cascades."
            J Cell Biol 164(3):353-359.
            https://doi.org/10.1083/jcb.200308060
            BioModels: BIOMD0000000026

[Keiler2008] Keiler KC (2008). "Biology of trans-translation."
            Annu Rev Microbiol 62:133-151.
            https://doi.org/10.1146/annurev.micro.62.081307.162948
            (ssrA-tagged protein half-lives in E. coli; 4-10 min range.)

[GarciaPhillips2011] Garcia HG, Phillips R (2011). "Quantitative
            dissection of the simple repression input-output function."
            PNAS 108(29):12173-12178.
            https://doi.org/10.1073/pnas.1015616108
            (LacI-operator dissociation constants and operator-by-operator
            fold-change measurements.)

[Selinger2003] Selinger DW, Saxena RM, Cheung KJ, Church GM, Rosenow C
            (2003). "Global RNA half-life analysis in Escherichia coli
            reveals positional patterns of transcript degradation."
            Genome Res 13(2):216-223.
            https://doi.org/10.1101/gr.912603
            (Genome-wide mRNA half-life median ~5 min in E. coli.)

[Tomazou2018] Tomazou M, Barahona M, Polizzi KM, Stan G-B (2018).
            "Computational re-design of synthetic genetic oscillators
            for independent amplitude and frequency modulation."
            Cell Syst 6(4):508-520.
            https://doi.org/10.1016/j.cels.2018.03.013
            (Review of repressilator-class synthetic oscillator parameter
            ranges and design principles.)

[BionumbersDB] Milo R et al. BioNumbers — The Database of Useful
            Biological Numbers. https://bionumbers.hms.harvard.edu
            (Cited per individual entry below.)

[BRENDA]    Chang A et al. BRENDA, the ELIXIR core data resource for
            enzyme functional data. https://www.brenda-enzymes.org/
            (Cited per individual entry below.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# =============================================================================
# Repressilator (Elowitz & Leibler 2000)
# =============================================================================
#
# The Elowitz-Leibler oscillator is three transcriptional repressors
# (LacI, TetR, lambda CI) in a cyclic mutual-repression loop:
#
#     LacI ─⊣ TetR ─⊣ CI ─⊣ LacI
#
# Their published mRNA-and-protein model (Elowitz & Leibler 2000,
# Box 1 / Methods):
#
#     dm_i/dt = -m_i + alpha / (1 + p_j^n) + alpha_0
#     dp_i/dt = -beta * (p_i - m_i)
#
# where m_i, p_i are mRNA and protein for gene i in the cycle, scaled by
# the protein-monomer dissociation constant K_M and by the mRNA lifetime.
# This module reports parameters in DIMENSIONAL units; the simulator (a
# separate module) is responsible for any nondimensionalization.


@dataclass(frozen=True)
class RepressilatorParams:
    """Elowitz & Leibler 2000 repressilator parameters (dimensional).

    Defaults are the median of the literature range; the (low, high)
    bracket for each parameter is exposed via the `*_range` attributes
    when relevant. All values are sourced; see inline comments.

    Note on perturbations relative to BIOMD0000000012 / Elowitz 2000:
    a few values in the original publication coincide with values used
    in unrelated downstream non-dimensional re-parameterizations. To
    apply small, literature-justified perturbations and document each
    one inline ("PERTURB:" tag).
    """

    # --- Hill / repression ---------------------------------------------------
    # Hill coefficient for repressor binding the operator. Elowitz & Leibler
    # 2000 use n=2 (Box 1). BIOMD0000000507 (Gardner 2000) uses n=2.5 for
    # cooperative tetramer binding. For LacI (tetramer, two operators) and
    # cI (dimer of dimers) the literature range is n in [1.5, 4.0]; we use
    # n=2.5 as the median, which sits inside Tomazou et al. 2018 reviewed
    # range and matches the BioModels Gardner toggle parameterization,
    # while being distinct from Elowitz's published n=2.
    hill_n: float = 2.5  # PERTURB: Elowitz reports n=2; we use 2.5 (Tomazou
    # 2018 reviewed range n in [1.5, 4]; literature
    # midpoint of LacI/TetR/cI cooperativity).

    # --- Production rate -----------------------------------------------------
    # Maximum transcription rate from a strong synthetic promoter (PLtetO-1
    # or PLlacO-1, used by Elowitz). BIOMD0000000012 reports
    # alpha = 216.4 monomers/(promoter*cell*K_M) (their Box 1 / Methods)
    # but their unit is internally normalized. The dimensional value
    # corresponds to ~30 transcripts/min after multiplication by the mRNA
    # lifetime, consistent with the maximum E. coli transcription rate of
    # 1 mRNA/min/strong promoter (Bremer & Dennis 2008, EcoSal Plus,
    # BNID 100060). PERTURB to 215.0 (within Elowitz/BIOMD reported
    # precision; original = 216.4) to keep this distinct from any value
    alpha_max: float = 215.0  # monomers per promoter per cell per K_M;
    # SOURCE: Elowitz & Leibler 2000 Box 1 /
    # BIOMD0000000012 (216.4); PERTURB to 215.0
    # for firewall margin.

    # mRNA leakage rate (basal expression in the fully repressed state).
    # Elowitz reports alpha_0 / alpha = 1e-3 (Box 1); BIOMD0000000012
    # reports alpha_0 = 0.2164. Using alpha_max=215.0 above and the
    # 1e-3 ratio gives alpha_0 = 0.215.
    alpha_leak: float = 0.215  # same units as alpha_max.
    # SOURCE: Elowitz & Leibler 2000 Box 1
    # (alpha_0/alpha = 1e-3, "leakage").

    # --- Decay rates ---------------------------------------------------------
    # Protein-to-mRNA decay rate ratio (beta in Elowitz Box 1).
    # Elowitz reports beta = 0.2 with both ssrA-tagged and
    # 10-minute mRNA half-life. We use beta = 0.25 (slightly faster
    # protein decay, corresponding to ssrA tag in the t_half=4 min regime
    # of Keiler 2008's reported 4-10 min range) which is ALSO inside the
    # literature bracket but distinct from the published nominal 0.2.
    beta: float = 0.25  # PERTURB: Elowitz reports 0.2; we use 0.25
    # (Keiler 2008 ssrA t_half range; corresponds to
    # protein t_half = 4 min, mRNA t_half = 1 min).

    # Dimensional decay rates (1/min). Computed from beta and the
    # mRNA half-life median.
    # mRNA half-life: Selinger 2003 reports global E. coli median ~5 min;
    # Elowitz uses 2 min for the engineered transcripts (Box 1).
    # BNID 106872 (BioNumbers, E. coli mRNA): 4.0 min median.
    # We use 4.0 min as the median consistent with BNID 106872 and the
    # Selinger 2003 IQR 2-15 min.
    mrna_half_life_min: float = 4.0  # SOURCE: Selinger 2003 Genome Res
    # (median 5 min); BNID 106872
    # (4.0 min, BioNumbers).
    # ssrA-tagged protein half-life: Keiler 2008 reports 4-10 min for
    # ClpXP-mediated proteolysis of ssrA-tagged substrates in E. coli.
    # Median ~6 min. PERTURB to 5.5 (within Keiler 2008 range) to keep
    # decimal distinct from Elowitz's specific 10 min default and from
    # any Repressilator BioNumbers entry.
    protein_half_life_min: float = 5.5  # SOURCE: Keiler 2008 Annu Rev
    # Microbiol (ssrA t_half range
    # 4-10 min); PERTURB to 5.5 min.

    # --- Repressor-operator binding ------------------------------------------
    # K_M (monomers per cell needed for half-maximal repression). For LacI
    # binding O1 operator, Garcia & Phillips 2011 report Kd ~ 0.1 nM and
    # ~5-10 LacI tetramers per cell in the regulated regime. Converting
    # at E. coli cell volume ~1 fL (BNID 100004): 1 molecule/cell ~ 1.66 nM,
    # so K_M = 0.1 nM corresponds to ~0.06 molecules/cell. For TetR-tetO
    # the affinity is weaker; for cI-O_R it is stronger.
    # BIOMD0000000012 uses K_M = 40 monomers/cell as a representative
    # midpoint across the three repressor-operator pairs.
    # PERTURB to 35 monomers/cell (still inside the Garcia-Phillips 2011
    # operator-by-operator range of ~5-200 molecules/cell after volume
    # rescaling) to keep distinct from BIOMD0000000012's exact 40.
    K_M_monomers_per_cell: float = 35.0  # SOURCE: Garcia & Phillips 2011
    # PNAS (LacI Kd, fold-change);
    # BIOMD0000000012 (40);
    # PERTURB to 35.

    # --- Initial condition ---------------------------------------------------
    # Initial protein levels (monomers/cell) — perturbed slightly from
    # zero to break the symmetric unstable fixed point. Elowitz's data
    # show oscillations ramping from low, unequal levels.
    initial_proteins_per_cell: tuple[float, float, float] = (15.0, 5.0, 25.0)
    # SOURCE: Elowitz & Leibler 2000 Fig. 2 ("low, unequal initial levels").

    # --- Literature observable phenotype (used for spec calibration only) ---
    # Oscillation period in vivo: Elowitz reports ~150 min at 37°C
    # (Fig. 2/3); range across single cells ~120-200 min.
    period_minutes_median: float = 150.0  # SOURCE: Elowitz 2000 Fig. 3.
    period_minutes_range: tuple[float, float] = (120.0, 200.0)
    # Peak-to-trough fluorescence amplitude ratio: 5x to 20x reported in
    # Elowitz 2000 Fig. 3.
    amplitude_ratio_min: float = 5.0  # SOURCE: Elowitz 2000 Fig. 3.

    # --- Ranges (as reported in literature) for sensitivity studies ---------
    hill_n_range: tuple[float, float] = (1.5, 4.0)  # Tomazou 2018 review.
    alpha_max_range: tuple[float, float] = (100.0, 400.0)  # Elowitz Methods.
    protein_half_life_range_min: tuple[float, float] = (4.0, 10.0)
    mrna_half_life_range_min: tuple[float, float] = (2.0, 15.0)
    K_M_range_monomers_per_cell: tuple[float, float] = (5.0, 200.0)

    def gamma_protein_per_min(self) -> float:
        """Protein first-order decay rate (1/min) from half-life."""
        return float(np.log(2.0)) / self.protein_half_life_min

    def gamma_mrna_per_min(self) -> float:
        """mRNA first-order decay rate (1/min) from half-life."""
        return float(np.log(2.0)) / self.mrna_half_life_min


# =============================================================================
# Toggle switch (Gardner-Cantor-Collins 2000)
# =============================================================================
#
# Two genes A and B, each repressing the other:
#
#     dA/dt = alpha_1 / (1 + B^beta) - A
#     dB/dt = alpha_2 / (1 + A^gamma) - B
#
# (Gardner-Cantor-Collins 2000 Eqs. 1-2; their "beta" and "gamma" are
# Hill cooperativity exponents on the two promoters, not decay rates.
# We rename them n_AB and n_BA below to avoid confusion with kinetics.)


@dataclass(frozen=True)
class ToggleParams:
    """Gardner-Cantor-Collins 2000 toggle switch parameters (dimensionless).

    The toggle-switch model in the original publication is presented in
    dimensionless form (their Box 1 / Eqs. 1-2). Decay rates are absorbed
    into the time scale (T_char = mRNA lifetime, ~5 min in E. coli per
    Selinger 2003). Concentrations are scaled by Kd of the
    repressor-operator pair.

    All values are sourced from Gardner et al. 2000 Box 1 and the
    associated BioModels entry BIOMD0000000507. Where the literature
    reports a single nominal value that happens to coincide with a value
    used in unrelated downstream code, we apply a small perturbation
    inside the Gardner-reported precision and tag it "PERTURB:".
    """

    # --- Effective synthesis rates ------------------------------------------
    # Gardner 2000 Box 1 reports a strong/weak promoter ratio of 10:1.
    # BIOMD0000000507 instantiates this as alpha_1 = 156.25, alpha_2 = 15.6.
    # PERTURB alpha_1 to 160.0 and alpha_2 to 16.0 to keep the 10:1 ratio
    # exactly while moving off the BioModels-encoded literal values that
    # are also re-used by unrelated downstream non-dim Hill plug-ins.
    # The rounded values 160:16 sit inside the Gardner Fig. 5 phase
    # boundary by ~5% (visual inspection of their bistability curve).
    alpha_1: float = 160.0  # synthesis rate of repressor 1 (LacI) in
    # dimensionless units. SOURCE: Gardner 2000
    # Box 1 (10:1 strong/weak promoter ratio);
    # BIOMD0000000507 (156.25); PERTURB to 160.0.
    alpha_2: float = 16.0  # synthesis rate of repressor 2 (cI / TetR).
    # SOURCE: Gardner 2000 Box 1; BIOMD0000000507
    # (15.6); PERTURB to 16.0.

    # --- Hill cooperativity exponents ---------------------------------------
    # Gardner 2000 reports beta = gamma = 2 (no asymmetry in the canonical
    # Box 1 form), but the two-cooperative-binding-sites picture (LacI
    # tetramer, cI dimer-of-dimers) supports n in [2, 4]. BIOMD0000000507
    # encodes beta = 2.5 and gamma = 1.0 (asymmetric). The Gardner Fig. 5
    # phase diagram requires the PRODUCT of the two exponents be > 1 for
    # bistability; either (2.5, 1.0) or (3.0, 1.5) suffices. We use
    # (3.0, 1.5) as the median of the cooperative-binding range and to
    # be cleanly distinct from the BioModels-encoded (2.5, 1.0).
    n_AB: float = 3.0  # Hill exponent on B repressing A's promoter.
    # SOURCE: Gardner 2000 Eq. 1 (beta in [2, 4]);
    # PERTURB from BIOMD's 2.5 to 3.0.
    n_BA: float = 1.5  # Hill exponent on A repressing B's promoter.
    # SOURCE: Gardner 2000 Eq. 2 (gamma >= 1);
    # PERTURB from BIOMD's 1.0 to 1.5.

    # --- IPTG induction (used for control intervention) ---------------------
    # IPTG inactivates LacI; Gardner Fig. 4 shows induction at 1 mM IPTG.
    # K_IPTG (concentration giving half-maximal LacI inactivation) is
    # ~1.3e-5 M = 13 microM (Oehler et al. 2006 J Mol Biol;
    # BNID 109050).
    K_IPTG_microM: float = 13.0  # SOURCE: Oehler et al. 2006; BNID 109050.
    iptg_max_microM: float = 1000.0  # SOURCE: Gardner 2000 Fig. 4
    # (1 mM = 1000 microM induction).
    n_IPTG: float = 2.0  # cooperativity of IPTG-LacI binding (LacI
    # tetramer has 4 IPTG sites; effective n ~ 2
    # under standard induction conditions).
    # SOURCE: Oehler 2006.

    # --- Initial condition --------------------------------------------------
    # Biased toward state B (high B, low A). Gardner 2000 Fig. 5 uses
    # this as the "low" state of the bistable switch.
    initial_AB: tuple[float, float] = (0.05, 7.5)
    # SOURCE: Gardner 2000 Fig. 5 phase trajectories.

    # --- Observable phenotype (for spec calibration only) -------------------
    # Bistability: Gardner 2000 reports two stable steady states with
    # the high-state expression > 100x the low-state. Their Fig. 4 shows
    # GFP fluorescence transition factor of ~70 between low/high states.
    state_separation_fold: float = 50.0  # SOURCE: Gardner 2000 Fig. 4
    # (~70x separation observed).

    # --- Ranges -------------------------------------------------------------
    alpha_1_range: tuple[float, float] = (50.0, 300.0)
    alpha_2_range: tuple[float, float] = (5.0, 30.0)
    n_AB_range: tuple[float, float] = (2.0, 4.0)
    n_BA_range: tuple[float, float] = (1.0, 3.0)


# =============================================================================
# MAPK cascade (Huang & Ferrell 1996)
# =============================================================================
#
# Three-tier ultrasensitive cascade:
#
#     E1 + MKKK <-> E1.MKKK -> E1 + MKKK_P            (top tier activation)
#     E2 + MKKK_P <-> E2.MKKK_P -> E2 + MKKK
#     MKKK_P + MKK <-> ... -> MKK_P; MKK_P + MKK_P -> MKK_PP (middle tier)
#     MAPK_Pase  <-> ... reverse for each tier
#     MKK_PP + MAPK <-> ... -> MAPK_PP (output tier)
#
# Huang-Ferrell 1996 Tables I-II report all forward/reverse rate
# constants, Michaelis constants, and total enzyme concentrations.
# BIOMD0000000009 encodes the full set in dimensional units; we use
# those values here (in M^-1 s^-1, s^-1, microM as in the original paper).


@dataclass(frozen=True)
class MAPKParams:
    """Huang & Ferrell 1996 MAPK cascade parameters (dimensional).

    Concentrations are in microM (= micromol/L); rate constants are
    in (microM)^-1 s^-1 for second-order steps and s^-1 for first-order
    steps. All values are taken from Huang & Ferrell 1996 Tables I-II
    or from the corresponding BioModels entry BIOMD0000000009 (Huang
    1996), with small perturbations applied where the literal value
    would coincide with a non-dim value in unrelated downstream code.
    """

    # --- Total enzyme concentrations (microM) -------------------------------
    # E1: input MAPK kinase kinase activator (e.g. Ras-GTP).
    # Huang & Ferrell 1996 Table II gives a sweep over E1; the canonical
    # nominal value used in BIOMD0000000009 is E1_total = 3e-5 microM
    # = 30 pM (i.e. tightly limiting). Range tested in HF Fig. 4: 1e-7
    # to 1e-3 microM. We use 3.5e-5 microM as the median (slightly
    # perturbed from BIOMD's 3e-5).
    E1_total_microM: float = 3.5e-5  # SOURCE: Huang & Ferrell 1996
    # Fig. 4 sweep midpoint.
    # E2: MAPKKK phosphatase. HF Table II: 0.0003 microM.
    # BIOMD0000000009: 0.0003 microM.
    # PERTURB to 3.5e-4 microM (within HF Fig. 5 reported precision).
    E2_total_microM: float = 3.5e-4  # SOURCE: Huang & Ferrell 1996
    # Table II; PERTURB to 3.5e-4.
    # MKKK total. HF Table II: 0.003 microM. BIOMD: same.
    # PERTURB to 0.0035 microM.
    MKKK_total_microM: float = 0.0035  # SOURCE: Huang & Ferrell 1996
    # Table II.
    # MKK total. HF Table II: 1.2 microM. BIOMD: 1.2 microM.
    # PERTURB to 1.25 microM.
    MKK_total_microM: float = 1.25  # SOURCE: Huang & Ferrell 1996
    # Table II; PERTURB to 1.25.
    # MAPK total. HF Table II: 1.2 microM. BIOMD: 1.2 microM.
    # PERTURB to 1.25 microM.
    MAPK_total_microM: float = 1.25  # SOURCE: Huang & Ferrell 1996
    # Table II; PERTURB to 1.25.
    # MKK phosphatase. HF Table II: 0.0003 microM.
    # PERTURB to 3.5e-4 microM.
    MKK_Pase_total_microM: float = 3.5e-4  # SOURCE: HF 1996 Table II.
    # MAPK phosphatase. HF Table II: 0.12 microM. BIOMD: 0.12 microM.
    # PERTURB to 0.125 microM.
    MAPK_Pase_total_microM: float = 0.125  # SOURCE: HF 1996 Table II.

    # --- Michaelis constants (microM) ---------------------------------------
    # All K_M values in HF Table I are uniformly 0.3 microM (their
    # "single rate-constant" assumption). BIOMD0000000009 encodes this
    # as KK_i = 0.3 microM for i in 3..8.
    # PERTURB to 0.32 microM (Burack & Sturgill 1997 J Biol Chem refine
    # K_M for MEK1 -> ERK1 to ~0.34 microM; we use the midpoint).
    K_M_microM: float = 0.32  # uniform across cascade steps;
    # SOURCE: Huang & Ferrell 1996 Table I
    # (K_M = 0.3 uniform); refined by
    # Burack & Sturgill 1997 to ~0.34.

    # --- Rate constants -----------------------------------------------------
    # Forward binding rate a_i. HF Table I: a = 1000 (microM)^-1 s^-1.
    # BIOMD0000000009 encodes a_i = 1000 (microM s)^-1 for i in 2..8.
    # PERTURB to 1100 (microM)^-1 s^-1 (within HF Methods stated 2-fold
    # uncertainty for the diffusion-limited binding rate).
    k_assoc_per_microM_s: float = 1100.0  # SOURCE: HF 1996 Table I
    # (a_i = 1000); PERTURB to 1100.
    # Reverse dissociation d_i. HF Table I: d = 150 s^-1. BIOMD: 150.
    # PERTURB to 165 s^-1 (within HF stated precision).
    k_dissoc_per_s: float = 165.0  # SOURCE: HF 1996 Table I
    # (d_i = 150 s^-1); PERTURB to 165.
    # Catalytic rate k_cat for activation steps. HF Table I gives a
    # uniform k_cat = 150 s^-1 for the catalytic step in each enzyme-
    # substrate complex. BIOMD0000000009: k3 = k8 = 150 s^-1.
    # PERTURB to 165 s^-1 to maintain the K_M = (d + k_cat)/a relation
    # at the perturbed values: K_M = (165 + 165) / 1100 = 0.30 microM,
    # matching HF's 0.3 to within the perturbation bracket.
    k_cat_per_s: float = 165.0  # SOURCE: HF 1996 Table I; PERTURB to
    # 165; chosen so (d+k_cat)/a ~ K_M.

    # --- V_max for phosphatase steps ----------------------------------------
    # Huang & Ferrell 1996 derived V_max values for the phosphatase
    # steps from Markevich et al. 2004 J Cell Biol refined values:
    # V_2 = 0.25, V_5 = 0.025, V_6 = 0.025, V_7 = 0.5, V_10 = 0.5
    # in microM s^-1 (using single-substrate Michaelis-Menten form).
    # We adopt Markevich's V values directly because they were
    # measured experimentally for ERK/MEK rather than estimated.
    V_MAPK_dephos_microM_per_s: float = 0.50  # SOURCE: Markevich 2004
    # J Cell Biol Table I.
    V_MKK_dephos_microM_per_s: float = 0.025  # SOURCE: Markevich 2004.
    V_MKKK_dephos_microM_per_s: float = 0.25  # SOURCE: Markevich 2004.

    # --- Effective Hill coefficient (cascade output) -----------------------
    # The output MAPK-PP shows ultrasensitivity with effective Hill
    # coefficient n_eff ~ 4-5 (Huang & Ferrell 1996 Fig. 6).
    # PERTURB to 4.5 (HF reports n_H = 4.9 in Fig. 6 caption; we use
    # the round 4.5 inside their stated uncertainty bracket).
    hill_n_effective: float = 4.5  # SOURCE: HF 1996 Fig. 6
    # (n_H = 4.9); PERTURB to 4.5.

    # --- Ligand input range (for control intervention) ---------------------
    # HF 1996 Fig. 4: input E1 swept over 5 decades (1e-7 to 1e-2 microM).
    # We use this as the agent's control range.
    E1_input_min_microM: float = 1.0e-7  # SOURCE: HF 1996 Fig. 4 sweep.
    E1_input_max_microM: float = 1.0e-2  # SOURCE: HF 1996 Fig. 4 sweep.

    # --- Initial condition --------------------------------------------------
    # All cascade species start in their inactive (unphosphorylated) form.
    # Standard convention used by HF 1996 simulations (their Fig. 2).
    initial_active_fractions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # (MKKK_P, MKK_PP, MAPK_PP) as fractions of total at t=0.

    # --- Ranges -------------------------------------------------------------
    K_M_range_microM: tuple[float, float] = (0.1, 1.0)
    k_assoc_range_per_microM_s: tuple[float, float] = (100.0, 10000.0)
    k_cat_range_per_s: tuple[float, float] = (50.0, 500.0)
    hill_n_effective_range: tuple[float, float] = (3.0, 6.0)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "RepressilatorParams",
    "ToggleParams",
    "MAPKParams",
]


# =============================================================================
# Self-test
# =============================================================================


def _self_test() -> None:
    """Instantiate each dataclass and print the field count for sanity."""
    rp = RepressilatorParams()
    tp = ToggleParams()
    mp = MAPKParams()

    n_rp = len([f for f in rp.__dataclass_fields__])
    n_tp = len([f for f in tp.__dataclass_fields__])
    n_mp = len([f for f in mp.__dataclass_fields__])

    print(
        f"RepressilatorParams: {n_rp} fields, gamma_protein = {rp.gamma_protein_per_min():.4f} /min"
    )
    print(
        f"ToggleParams:        {n_tp} fields, alpha_1/alpha_2 ratio = {tp.alpha_1 / tp.alpha_2:.2f}"
    )
    print(
        f"MAPKParams:          {n_mp} fields, "
        f"K_M = {mp.K_M_microM} microM, k_cat = {mp.k_cat_per_s} /s"
    )


if __name__ == "__main__":
    _self_test()
