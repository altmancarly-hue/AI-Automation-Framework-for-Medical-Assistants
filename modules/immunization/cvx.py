"""CVX codes, antigen components, and historical trade-name mappings.

WHY this is pure data with no model anywhere near it:

A CVX code is an identifier from a published CDC table. Asking a language model
what CVX 110 contains is asking it to recall a lookup table from training data,
which it will do confidently and occasionally wrongly. The table is small, it is
public, and it is the input to a clinical decision. It goes in a dict.

The one genuinely hard thing here is **combination vaccines**. A dose of
Pediarix (CVX 110) is simultaneously a DTaP dose, a HepB dose and an IPV dose.
If the practice chart records "Pediarix" and the state registry records three
separate component rows on the same date, a naive reconciler sees one dose on
one side and three on the other and reports a discrepancy that does not exist.
Worse, a forecaster that does not expand components will believe the child is
two DTaP doses behind and generate a recall for a vaccine they have had.

So every dose is normalised to its **antigen components** before anything else
looks at it. `expand()` is the function the rest of the module is built on.

Coverage note, stated plainly: this table covers the routine childhood and
adolescent schedule the practice actually administers, plus the combination
products and historical trade names most likely to appear in a transferred-in
record. It is not the complete CVX set. `unknown_codes()` reports anything a
record contains that this table does not know, and the reconciler routes those
to a human rather than guessing -- an unknown code must never be silently
dropped, because a dropped dose looks exactly like a missing dose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

__all__ = [
    "Antigen",
    "VaccineProduct",
    "CVX",
    "expand",
    "components_for",
    "antigens_for",
    "product_for",
    "is_known",
    "unknown_codes",
    "same_antigen_set",
    "shares_any_antigen",
    "normalise_code",
    "TRADE_NAMES",
]


class Antigen:
    """The unit the forecaster reasons about. One antigen, one series."""

    DTAP = "dtap"
    TDAP = "tdap"
    TD = "td"
    HEPB = "hepb"
    HEPA = "hepa"
    IPV = "ipv"
    HIB = "hib"
    PCV = "pcv"
    PPSV = "ppsv"
    MMR = "mmr"
    VAR = "var"
    ROTA = "rota"
    HPV = "hpv"
    MENACWY = "menacwy"
    MENB = "menb"
    INFLUENZA = "influenza"
    COVID = "covid"
    RSV = "rsv"
    ALL = (
        DTAP, TDAP, TD, HEPB, HEPA, IPV, HIB, PCV, PPSV, MMR, VAR, ROTA,
        HPV, MENACWY, MENB, INFLUENZA, COVID, RSV,
    )


@dataclass(frozen=True)
class VaccineProduct:
    """One row of the CVX table, decomposed into the antigens it delivers."""

    code: str
    short_name: str
    description: str
    components: tuple[str, ...]
    #: Rotavirus and Hib series length depends on the product administered.
    #: Recorded here because the forecaster cannot infer it from the antigen.
    series_variant: str | None = None
    inactive: bool = False

    @property
    def is_combination(self) -> bool:
        return len(self.components) > 1


def _p(
    code: str,
    short_name: str,
    description: str,
    components: Iterable[str],
    *,
    series_variant: str | None = None,
    inactive: bool = False,
) -> tuple[str, VaccineProduct]:
    return code, VaccineProduct(
        code=code,
        short_name=short_name,
        description=description,
        components=tuple(components),
        series_variant=series_variant,
        inactive=inactive,
    )


#: CVX code -> product. Codes are strings because CDC publishes them zero-padded
#: in some feeds and bare in others; `normalise_code` reconciles the two.
CVX: Mapping[str, VaccineProduct] = dict(
    [
        # -- single antigen ------------------------------------------------
        _p("08", "HepB-peds", "hepatitis B, pediatric/adolescent dosage", [Antigen.HEPB]),
        _p("42", "HepB-adol/adult", "hepatitis B, adolescent/high risk infant", [Antigen.HEPB]),
        _p("43", "HepB-adult", "hepatitis B, adult dosage", [Antigen.HEPB]),
        _p("45", "HepB-unspec", "hepatitis B, unspecified formulation", [Antigen.HEPB]),
        _p("189", "HepB-CpG", "hepatitis B, CpG adjuvanted (Heplisav-B)", [Antigen.HEPB]),
        _p("20", "DTaP", "diphtheria, tetanus toxoids and acellular pertussis", [Antigen.DTAP]),
        _p("106", "DTaP-5pertussis", "DTaP, 5 pertussis antigens", [Antigen.DTAP]),
        _p("107", "DTaP-unspec", "DTaP, unspecified formulation", [Antigen.DTAP]),
        _p("115", "Tdap", "tetanus, diphtheria, acellular pertussis", [Antigen.TDAP]),
        _p("09", "Td-adult", "Td (adult), adsorbed", [Antigen.TD]),
        _p("113", "Td-preservative-free", "Td (adult), preservative free", [Antigen.TD]),
        _p("10", "IPV", "poliovirus vaccine, inactivated", [Antigen.IPV]),
        _p("89", "polio-unspec", "poliovirus vaccine, unspecified", [Antigen.IPV]),
        _p("02", "OPV", "poliovirus vaccine, live, oral", [Antigen.IPV], inactive=True),
        _p("49", "Hib-PRP-OMP", "Hib, PRP-OMP conjugate (PedvaxHIB)", [Antigen.HIB],
           series_variant="prp_omp"),
        _p("48", "Hib-PRP-T", "Hib, PRP-T conjugate (ActHIB/Hiberix)", [Antigen.HIB],
           series_variant="prp_t"),
        _p("17", "Hib-unspec", "Hib conjugate, unspecified formulation", [Antigen.HIB]),
        _p("133", "PCV13", "pneumococcal conjugate 13-valent", [Antigen.PCV]),
        _p("215", "PCV15", "pneumococcal conjugate 15-valent", [Antigen.PCV]),
        _p("216", "PCV20", "pneumococcal conjugate 20-valent", [Antigen.PCV]),
        _p("152", "PCV-unspec", "pneumococcal conjugate, unspecified", [Antigen.PCV]),
        _p("33", "PPSV23", "pneumococcal polysaccharide 23-valent", [Antigen.PPSV]),
        _p("03", "MMR", "measles, mumps and rubella", [Antigen.MMR]),
        _p("21", "VAR", "varicella", [Antigen.VAR]),
        _p("83", "HepA-peds", "hepatitis A, pediatric/adolescent, 2 dose", [Antigen.HEPA]),
        _p("85", "HepA-unspec", "hepatitis A, unspecified formulation", [Antigen.HEPA]),
        _p("119", "RV1", "rotavirus, monovalent (Rotarix)", [Antigen.ROTA],
           series_variant="rv1"),
        _p("116", "RV5", "rotavirus, pentavalent (RotaTeq)", [Antigen.ROTA],
           series_variant="rv5"),
        _p("122", "RV-unspec", "rotavirus, unspecified formulation", [Antigen.ROTA]),
        _p("62", "HPV4", "HPV, quadrivalent (Gardasil)", [Antigen.HPV], inactive=True),
        _p("165", "HPV9", "HPV, 9-valent (Gardasil 9)", [Antigen.HPV]),
        _p("137", "HPV-unspec", "HPV, unspecified formulation", [Antigen.HPV]),
        _p("114", "MenACWY-D", "meningococcal MCV4P (Menactra)", [Antigen.MENACWY]),
        _p("136", "MenACWY-CRM", "meningococcal MCV4O (Menveo)", [Antigen.MENACWY]),
        _p("203", "MenACWY-TT", "meningococcal MenACWY-TT (MenQuadfi)", [Antigen.MENACWY]),
        _p("162", "MenB-4C", "meningococcal B, OMV (Bexsero)", [Antigen.MENB],
           series_variant="menb_4c"),
        _p("163", "MenB-FHbp", "meningococcal B, recombinant (Trumenba)", [Antigen.MENB],
           series_variant="menb_fhbp"),
        _p("150", "IIV4", "influenza, quadrivalent, preservative free", [Antigen.INFLUENZA]),
        _p("158", "IIV4", "influenza, injectable, quadrivalent", [Antigen.INFLUENZA]),
        _p("161", "IIV-peds", "influenza, injectable, quadrivalent, peds", [Antigen.INFLUENZA]),
        _p("149", "LAIV4", "influenza, live, quadrivalent, intranasal", [Antigen.INFLUENZA]),
        _p("171", "COVID-mRNA", "COVID-19, mRNA", [Antigen.COVID]),
        _p("312", "COVID-2024", "COVID-19, mRNA, 2024-2025 formula", [Antigen.COVID]),
        _p("306", "RSV-mAb", "nirsevimab (Beyfortus)", [Antigen.RSV]),
        # -- combinations: the reason this module exists -------------------
        _p("110", "Pediarix", "DTaP-HepB-IPV (Pediarix)",
           [Antigen.DTAP, Antigen.HEPB, Antigen.IPV]),
        _p("120", "Pentacel", "DTaP-IPV/Hib (Pentacel)",
           [Antigen.DTAP, Antigen.IPV, Antigen.HIB], series_variant="prp_t"),
        _p("146", "Vaxelis", "DTaP-IPV-Hib-HepB (Vaxelis)",
           [Antigen.DTAP, Antigen.IPV, Antigen.HIB, Antigen.HEPB], series_variant="prp_omp"),
        _p("130", "Kinrix/Quadracel", "DTaP-IPV (Kinrix, Quadracel)",
           [Antigen.DTAP, Antigen.IPV]),
        _p("50", "TriHIBit", "DTaP-Hib (TriHIBit)", [Antigen.DTAP, Antigen.HIB],
           series_variant="prp_t", inactive=True),
        _p("51", "Comvax", "Hib-HepB (Comvax)", [Antigen.HIB, Antigen.HEPB],
           series_variant="prp_omp", inactive=True),
        _p("94", "ProQuad", "MMRV (ProQuad)", [Antigen.MMR, Antigen.VAR]),
        _p("104", "Twinrix", "HepA-HepB (Twinrix)", [Antigen.HEPA, Antigen.HEPB]),
        _p("132", "DTaP-IPV-Hib-HepB", "DTaP-IPV-Hib-HepB, historical",
           [Antigen.DTAP, Antigen.IPV, Antigen.HIB, Antigen.HEPB], inactive=True),
        _p("197", "MenACWY-Hib", "MenACWY-TT/Hib (MenQuadfi combination)",
           [Antigen.MENACWY, Antigen.HIB]),
    ]
)


#: Free-text product names that turn up in transferred-in paper records and
#: registry comment fields. WHY a lookup and not a fuzzy matcher: a wrong guess
#: here credits a dose the child never had. Anything not on this list is an
#: unknown code and goes to a human.
TRADE_NAMES: Mapping[str, str] = {
    "pediarix": "110",
    "pentacel": "120",
    "vaxelis": "146",
    "kinrix": "130",
    "quadracel": "130",
    "infanrix": "20",
    "daptacel": "20",
    "tripedia": "20",
    "trihibit": "50",
    "comvax": "51",
    "proquad": "94",
    "mmr ii": "03",
    "mmrii": "03",
    "varivax": "21",
    "recombivax": "08",
    "engerix": "08",
    "engerix-b": "08",
    "heplisav": "189",
    "heplisav-b": "189",
    "havrix": "83",
    "vaqta": "83",
    "twinrix": "104",
    "ipol": "10",
    "acthib": "48",
    "hiberix": "48",
    "pedvaxhib": "49",
    "prevnar": "133",
    "prevnar 13": "133",
    "prevnar13": "133",
    "prevnar 20": "216",
    "vaxneuvance": "215",
    "pneumovax": "33",
    "pneumovax 23": "33",
    "rotarix": "119",
    "rotateq": "116",
    "gardasil": "62",
    "gardasil 9": "165",
    "gardasil9": "165",
    "menactra": "114",
    "menveo": "136",
    "menquadfi": "203",
    "bexsero": "162",
    "trumenba": "163",
    "boostrix": "115",
    "adacel": "115",
    "beyfortus": "306",
    "nirsevimab": "306",
}


def normalise_code(code: str | int | None) -> str | None:
    """Canonicalise a CVX code to the zero-padded two-digit-or-more string form.

    Registries emit "8", "08" and " 08 " for the same vaccine, and a dict lookup
    on the raw value silently misses two of the three. Trade names are accepted
    here too, because a paper record transcribed by a human is one of the main
    sources this module has to consume.
    """
    if code is None:
        return None
    text = str(code).strip()
    if not text:
        return None
    if text.isdigit():
        padded = text.lstrip("0") or "0"
        for candidate in (text, padded, padded.zfill(2), padded.zfill(3)):
            if candidate in CVX:
                return candidate
        return padded.zfill(2)
    lowered = text.lower().strip().rstrip(".")
    if lowered in TRADE_NAMES:
        return TRADE_NAMES[lowered]
    # "Pediarix (DTaP-HepB-IPV)" and similar -- take the leading product word.
    head = lowered.split("(")[0].strip()
    if head in TRADE_NAMES:
        return TRADE_NAMES[head]
    return text


def is_known(code: str | int | None) -> bool:
    return normalise_code(code) in CVX


def product_for(code: str | int | None) -> VaccineProduct | None:
    normalised = normalise_code(code)
    return CVX.get(normalised) if normalised else None


def components_for(code: str | int | None) -> tuple[str, ...]:
    """Antigens delivered by one CVX code. Empty tuple if the code is unknown.

    Callers MUST distinguish empty-because-unknown from empty-because-nothing;
    `is_known` is the check. Treating an unknown code as "no antigens" would
    make an unrecognised dose indistinguishable from a missing dose, and the
    forecaster would recall a child for a vaccine sitting in their chart.
    """
    product = product_for(code)
    return product.components if product else ()


def antigens_for(codes: Iterable[str | int]) -> set[str]:
    out: set[str] = set()
    for code in codes:
        out.update(components_for(code))
    return out


def unknown_codes(codes: Iterable[str | int]) -> list[str]:
    """Codes this table cannot resolve, in input order, de-duplicated."""
    seen: list[str] = []
    for code in codes:
        normalised = normalise_code(code)
        if normalised is not None and normalised not in CVX and normalised not in seen:
            seen.append(normalised)
    return seen


def expand(code: str | int | None) -> list[dict[str, str | None]]:
    """One administered product -> one record per antigen component.

    This is the normalisation the whole module depends on. A Pediarix dose
    becomes three antigen doses; the registry's three separate component rows
    on the same date become the same three antigen doses; and the two records
    then compare as equal instead of as a three-dose discrepancy.
    """
    product = product_for(code)
    if product is None:
        return []
    return [
        {
            "antigen": antigen,
            "cvx": product.code,
            "product": product.short_name,
            "series_variant": product.series_variant,
        }
        for antigen in product.components
    ]


def same_antigen_set(code_a: str | int | None, code_b: str | int | None) -> bool:
    """True when two codes deliver exactly the same antigens.

    This is the equivalence the deterministic matcher uses: CVX 08 and CVX 45
    are both "a HepB dose" and must reconcile, while CVX 110 (three antigens)
    and CVX 20 (one) must not be called equal even though they overlap.
    """
    a, b = components_for(code_a), components_for(code_b)
    return bool(a) and set(a) == set(b)


def shares_any_antigen(code_a: str | int | None, code_b: str | int | None) -> bool:
    """True when two codes overlap without being equivalent.

    Overlap without equality is the signature of the combination-vs-component
    recording difference, which is exactly the case that must go to
    adjudication rather than being resolved by a rule.
    """
    a, b = set(components_for(code_a)), set(components_for(code_b))
    return bool(a & b) and a != b
