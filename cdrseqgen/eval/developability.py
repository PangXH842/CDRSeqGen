"""Shared sequence developability & chemical liability indicators for CDR loops.

Single source of truth imported by benchmark/analysis/developability.py and available to
evaluation/generation workflows. Pure Python, zero external dependencies.

Indicators:
  * N-linked glycosylation motif: N[^P][ST]
  * Unpaired cysteines: odd count of 'C' (intra-loop disulfide requires even count)
  * Deamidation motifs: NG (high risk), NS / NT (moderate)
  * Aspartate isomerization motifs: DG (high risk), DS (moderate)
  * Oxidation-prone residues: M, W presence
  * Hydrophobic patch: consecutive run in {A, V, I, L, M, F, W, Y} of length >= 5
  * Net charge at pH 7.4: K + R + 0.1*H - D - E
  * Kyte-Doolittle GRAVY (mean hydropathy) and aromatic fraction (F, W, Y)
  * Major-liability-free rate: no N-glyc, no unpaired Cys, no NG, and no DG
"""
from __future__ import annotations

import re

NGLYC_RE = re.compile(r"N[^P][ST]")
HYDROPHOBIC_RUN_RE = re.compile(r"[AVILMFWY]+")

# Standard Kyte-Doolittle hydropathy scale (1982).
KYTE_DOOLITTLE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
AROMATIC = frozenset("FWY")

SCALAR_METRICS = (
    "major_liability_free_rate",
    "nglyc_rate",
    "unpaired_cys_rate",
    "deamidation_ng_rate",
    "deamidation_any_rate",
    "isomerization_dg_rate",
    "isomerization_any_rate",
    "oxidation_mw_rate",
    "hydrophobic_patch5_rate",
    "net_charge_ph74_mean",
    "gravy_mean",
    "aromatic_fraction_mean",
    "length_mean",
)


def score_sequence(seq: str) -> dict[str, float]:
    """Compute developability indicators for one non-empty CDR amino-acid string."""
    s = seq.strip().upper()
    if not s:
        raise ValueError("cannot compute developability on an empty sequence")
    n = len(s)
    nglyc = 1.0 if NGLYC_RE.search(s) else 0.0
    unpaired_cys = 1.0 if (s.count("C") % 2 == 1) else 0.0
    deam_ng = 1.0 if "NG" in s else 0.0
    deam_any = 1.0 if ("NG" in s or "NS" in s or "NT" in s) else 0.0
    iso_dg = 1.0 if "DG" in s else 0.0
    iso_any = 1.0 if ("DG" in s or "DS" in s) else 0.0
    ox_mw = 1.0 if ("M" in s or "W" in s) else 0.0
    max_hydro = max((len(m.group(0)) for m in HYDROPHOBIC_RUN_RE.finditer(s)), default=0)
    hydro5 = 1.0 if max_hydro >= 5 else 0.0
    major_free = 1.0 if not (nglyc or unpaired_cys or deam_ng or iso_dg) else 0.0
    net_charge = s.count("K") + s.count("R") + 0.1 * s.count("H") - s.count("D") - s.count("E")
    gravy = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in s) / n
    aromatic = sum(1 for aa in s if aa in AROMATIC) / n
    return {
        "major_liability_free_rate": major_free,
        "nglyc_rate": nglyc,
        "unpaired_cys_rate": unpaired_cys,
        "deamidation_ng_rate": deam_ng,
        "deamidation_any_rate": deam_any,
        "isomerization_dg_rate": iso_dg,
        "isomerization_any_rate": iso_any,
        "oxidation_mw_rate": ox_mw,
        "hydrophobic_patch5_rate": hydro5,
        "net_charge_ph74_mean": float(net_charge),
        "gravy_mean": float(gravy),
        "aromatic_fraction_mean": float(aromatic),
        "length_mean": float(n),
    }
