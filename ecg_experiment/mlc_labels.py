"""Frozen condition labels of MLC-01 over PTB-XL SCP, Challenge SNOMED CT and SPH AHA statements.

The rules are those of ``docs/mlc01-condition-dataset-protocol-2026-11-10.md``. Each record gets one of
three states per condition: 1.0 (positive), 0.0 (negative) or NaN (undefined). Undefined entries are
masked in a loss and left out of that condition's metrics; they are never negatives.

- Positive: the record lists a positive code of the condition. For PTB-XL any likelihood counts; the
  sensitivity flag marks positives that rest only on diagnostic statements with likelihood below 50.
- Undefined: the record lists only an ambiguous code of the condition; the source does not annotate the
  condition (CPSC 2018 outside its nine classes, Chapman for sinus arrhythmia, incomplete RBBB and
  fascicular blocks, Ningbo for atrial fibrillation, Georgia for ST depression); or the
  vocabulary cannot code it. The protocol's other undefined rules are ambiguous codes: left ventricular
  high voltage for left ventricular hypertrophy, SPH ST deviation without a direction modifier for ST
  depression, PTB-XL ``heart_axis`` LAD for left axis deviation, and PTB-XL subendocardial injury for
  myocardial infarction.
- Contradictory pairs on one record (complete with incomplete RBBB, LBBB with RBBB) make both undefined.
- SPH code 153 sets no condition of spaces B and C. In space A it counts for HYP, not STTC.
- Negative otherwise.

AHA specifications: ``"B"`` matches a statement with base code B and any modifier, ``"B+M"`` one that
carries modifier M, and ``"B="`` one without a modifier. SCP specifications ``"axis:X"`` match PTB-XL's
``heart_axis`` field. Space C conditions shared with space B have one definition, B's.
"""

from __future__ import annotations

import math

import numpy as np

from ecg_experiment import challenge_labels, challenge_splits, sph

SOURCES = ("ptbxl", *challenge_splits.SOURCES, "sph")
VOCABULARY = {"ptbxl": "scp", **dict.fromkeys(challenge_splits.SOURCES, "snomed"), "sph": "aha"}
FAMILIES = {"ptbxl": "ptbxl", **challenge_splits.FAMILIES, "sph": "sph"}
CPSC_2018_CLASSES = frozenset({"426783006", "164889003", "270492004", "164909002", "59118001", "284470004",
                               "164884008", "429622005", "164931005"})
LIKELIHOOD_THRESHOLD = 50.0
STATES = {1.0: "positive", 0.0: "negative"}


def codes(scp: tuple[str, ...] = (), snomed: tuple[str, ...] = (), aha: tuple[str, ...] = ()
          ) -> dict[str, frozenset[str]]:
    """
    Code specifications of one condition per vocabulary.

    Parameters
    ----------
    scp, snomed, aha : tuple[str, ...]
        PTB-XL SCP statements (or ``axis:X``), SNOMED CT codes and SPH AHA specifications.

    Returns
    -------
    dict[str, frozenset[str]]
        Keyed by ``scp``, ``snomed`` and ``aha``.
    """
    return {"scp": frozenset(scp), "snomed": frozenset(snomed), "aha": frozenset(aha)}


def condition(name: str, positive: dict[str, frozenset[str]],
              ambiguous: dict[str, frozenset[str]] | None = None) -> dict[str, object]:
    """
    One condition definition.

    Parameters
    ----------
    name : str
        Readable name.
    positive : dict[str, frozenset[str]]
        Output of ``codes``; codes that make a record positive.
    ambiguous : dict[str, frozenset[str]] | None
        Output of ``codes``; codes that make a record undefined when no positive code is listed.

    Returns
    -------
    dict[str, object]
        ``name``, ``positive`` and ``ambiguous``.
    """
    return {"name": name, "positive": positive, "ambiguous": ambiguous or codes()}


MI_SCP = ("IMI", "ASMI", "ILMI", "AMI", "ALMI", "LMI", "IPLMI", "IPMI", "PMI")
MI_SNOMED = ("164865005", "164867002", "54329005", "57054005")
INJURY_SCP = ("INJAS", "INJAL", "INJLA", "INJIN", "INJIL")
PREMATURE_UNKNOWN_SCP = ("PRC(S)", "BIGU", "TRIGU")
ST_NO_DIRECTION_AHA = ("145=", "146")

CONDITIONS: dict[str, dict[str, object]] = {
    "sinus_tachycardia": condition("Sinus tachycardia", codes(("STACH",), ("427084000",), ("21",))),
    "sinus_bradycardia": condition("Sinus bradycardia", codes(("SBRAD",), ("426177001",), ("22",)),
                                   codes(snomed=("426627000",))),
    "sinus_arrhythmia": condition("Sinus arrhythmia", codes(("SARRH",), ("427393009",), ("23",))),
    "af_or_flutter": condition("Atrial fibrillation or flutter", codes(
        ("AFIB", "AFLT"), ("164889003", "426749004", "282825002", "314208002", "164890007", "195080001"),
        ("50", "51"))),
    "premature_ventricular_complex": condition("Premature ventricular complex(es)", codes(
        ("PVC",), ("427172004", "17338001", "164884008", "11157007", "251180001", "251182009"), ("60",)),
        codes(PREMATURE_UNKNOWN_SCP, ("13640000",))),
    "first_degree_av_block": condition("First-degree AV block or prolonged PR", codes(
        ("1AVB", "LPR"), ("270492004", "164947007"), ("82",))),
    "complete_rbbb": condition("Complete right bundle branch block", codes(
        ("CRBBB",), ("713427006", "59118001"), ("106",)), codes(snomed=("6374002",))),
    "incomplete_rbbb": condition("Incomplete right bundle branch block", codes(
        ("IRBBB",), ("713426002",), ("105",))),
    "lafb": condition("Left anterior fascicular block", codes(("LAFB",), ("445118002",), ("101",))),
    "left_axis_deviation": condition("Left axis deviation", codes(("axis:ALAD",), ("39732003",), ("121",)),
                                     codes(scp=("axis:LAD",))),
    "lvh": condition("Left ventricular hypertrophy", codes(("LVH",), ("164873001",), ("142",)),
                     codes(("VCLVH",), ("55827005", "266249003", "370365005"))),
    "st_depression": condition("ST depression", codes(("STD_",), ("429622005", "704997005"), ("145+362",)),
                               codes(aha=ST_NO_DIRECTION_AHA)),
    "t_wave_abnormality": condition("T-wave abnormality", codes(
        ("NDT", "NT_", "TAB_", "INVT", "LOWT"), ("164934002", "59931005"), ("147", "146")),
        codes(snomed=("428750005", "55930002"))),
    "mi_any": condition("Myocardial infarction, any territory", codes(MI_SCP, MI_SNOMED,
                                                                      ("160", "161", "165", "166")),
                        codes(scp=INJURY_SCP)),
    "atrial_fibrillation": condition("Atrial fibrillation", codes(
        ("AFIB",), ("164889003", "426749004", "282825002", "314208002"), ("50",)),
        codes(snomed=("195080001",))),
    "atrial_flutter": condition("Atrial flutter", codes(("AFLT",), ("164890007",), ("51",)),
                                codes(snomed=("195080001",))),
    "supraventricular_tachycardia": condition("Supraventricular tachycardia", codes(
        ("SVTAC", "PSVT"), ("426761007", "67198005", "713422000", "233897008", "251166008", "426648003"),
        ("54",)), codes(scp=("SVARR",))),
    "premature_atrial_complex": condition("Premature atrial or supraventricular complex(es)", codes(
        ("PAC",), ("284470004", "63593006", "251170000", "251173003", "251168009"), ("30", "31")),
        codes(scp=PREMATURE_UNKNOWN_SCP)),
    "paced_rhythm": condition("Paced rhythm", codes(("PACE",), ("10370003", "251268003", "251266004"))),
    "complete_lbbb": condition("Complete left bundle branch block", codes(
        ("CLBBB",), ("733534002", "164909002"), ("104",)), codes(snomed=("6374002",))),
    "lpfb": condition("Left posterior fascicular block", codes(("LPFB",), ("445211001",), ("102",))),
    "nonspecific_ivcd": condition("Nonspecific intraventricular conduction delay", codes(
        ("IVCD",), ("698252002", "82226007"))),
    "preexcitation": condition("Ventricular pre-excitation", codes(
        ("WPW",), ("74390002", "195060002"), ("108",)), codes(snomed=("49578007",), aha=("80",))),
    "right_axis_deviation": condition("Right axis deviation", codes(("axis:ARAD",), ("47665007",), ("120",)),
                                      codes(scp=("axis:RAD",))),
    "low_qrs_voltage": condition("Low QRS voltage", codes(("LVOLT",), ("251146004",), ("125",))),
    "rvh": condition("Right ventricular hypertrophy", codes(("RVH",), ("89792004",), ("143",)),
                     codes(snomed=("266249003",), aha=("153",))),
    "left_atrial_enlargement": condition("Left atrial enlargement or abnormality", codes(
        ("LAO/LAE",), ("67741000119109", "446813000", "253352002"), ("140",)), codes(snomed=("195126007",))),
    "st_elevation": condition("ST elevation", codes(("STE_",), ("164931005",), ("145+363",)),
                              codes(aha=ST_NO_DIRECTION_AHA)),
    "long_qt": condition("Prolonged QT interval", codes(("LNGQT",), ("111975006",), ("148",))),
    "ischemic_st_t": condition("ST-T change labelled ischemic", codes(
        ("ISC_", "ISCAL", "ISCAS", "ISCLA", "ISCAN", "ISCIN", "ISCIL"),
        ("164861001", "413844008", "413444003", "426434006", "425419005", "425623009"))),
    "inferior_mi": condition("Inferior myocardial infarction", codes(("IMI", "ILMI", "IPLMI", "IPMI"),
                                                                     aha=("161",)),
                             codes(scp=("INJIN", "INJIL"))),
    "anterior_mi": condition("Anterior myocardial infarction", codes(
        ("AMI", "ASMI", "ALMI"), ("54329005",), ("160", "165", "166")),
        codes(("INJAS", "INJAL", "INJLA"), ("164865005", "164867002", "57054005"))),
}
SPACE_B = ("sinus_tachycardia", "sinus_bradycardia", "sinus_arrhythmia", "af_or_flutter",
           "premature_ventricular_complex", "first_degree_av_block", "complete_rbbb", "incomplete_rbbb",
           "lafb", "left_axis_deviation", "lvh", "st_depression", "t_wave_abnormality", "mi_any")
SPACE_C = ("sinus_tachycardia", "sinus_bradycardia", "sinus_arrhythmia", "atrial_fibrillation",
           "atrial_flutter", "supraventricular_tachycardia", "premature_atrial_complex",
           "premature_ventricular_complex", "paced_rhythm", "first_degree_av_block", "complete_rbbb",
           "incomplete_rbbb", "complete_lbbb", "lafb", "lpfb", "nonspecific_ivcd", "preexcitation",
           "left_axis_deviation", "right_axis_deviation", "low_qrs_voltage", "lvh", "rvh",
           "left_atrial_enlargement", "st_depression", "st_elevation", "t_wave_abnormality", "long_qt",
           "ischemic_st_t", "inferior_mi", "anterior_mi")
SPACE_A = ("MI", "STTC", "CD", "HYP")
NOT_ANNOTATED = {
    "sinus_arrhythmia": frozenset({"chapman_shaoxing"}),
    "incomplete_rbbb": frozenset({"chapman_shaoxing"}),
    "lafb": frozenset({"chapman_shaoxing"}),
    "lpfb": frozenset({"chapman_shaoxing"}),
    "atrial_fibrillation": frozenset({"ningbo"}),
    "st_depression": frozenset({"georgia"}),
}
CONTRADICTORY_PAIRS = (("complete_rbbb", "incomplete_rbbb"), ("complete_lbbb", "complete_rbbb"))
PARTIAL = {
    ("cpsc_2018", "af_or_flutter"): "atrial fibrillation is a CPSC 2018 class, atrial flutter is not",
    ("cpsc_2018", "STTC"): "only ST depression and ST elevation are CPSC 2018 classes",
    ("cpsc_2018", "CD"): "only first-degree AV block, LBBB and RBBB are CPSC 2018 classes",
}
SPACE_A_CODES = {
    "snomed": {name: frozenset(members) for name, members in challenge_labels.SUPERCLASSES.items()},
    "aha": {"MI": frozenset(sph.SUPERCLASSES["MI"]), "STTC": frozenset(sph.SUPERCLASSES["STTC"] - {"153"}),
            "CD": frozenset(sph.SUPERCLASSES["CD"]), "HYP": frozenset(sph.SUPERCLASSES["HYP"] | {"153"})},
}
LV_HIGH_VOLTAGE = {"scp": frozenset({"VCLVH"}), "snomed": frozenset({"55827005"}), "aha": frozenset()}
NORMAL_UNDEFINED = frozenset({"cpsc_2018_extra"})


def parse_snomed(text: str) -> frozenset[str]:
    """
    SNOMED CT codes of a Challenge ``Dx`` header field.

    Parameters
    ----------
    text : str
        Comma-separated codes, such as ``"164889003,59118001"``.

    Returns
    -------
    frozenset[str]
        The codes without surrounding spaces.
    """
    return frozenset(code.strip() for code in text.split(",") if code.strip())


def parse_aha(text: str) -> tuple[tuple[str, frozenset[str]], ...]:
    """
    SPH statements as base code and modifier set.

    Parameters
    ----------
    text : str
        ``AHA_Code`` field, such as ``"22;145+362"``.

    Returns
    -------
    tuple[tuple[str, frozenset[str]], ...]
        One ``(base, modifiers)`` pair per statement.
    """
    parts = [token.strip().split("+") for token in text.split(";") if token.strip()]
    return tuple((part[0], frozenset(part[1:])) for part in parts)


def ptbxl_codes(scp_codes: dict[str, float], heart_axis: str) -> dict[str, float]:
    """
    PTB-XL statements with the heart axis added as an ``axis:X`` pseudo-statement.

    Parameters
    ----------
    scp_codes : dict[str, float]
        Parsed ``scp_codes`` field: statement to likelihood.
    heart_axis : str
        ``heart_axis`` field; an empty string when missing.

    Returns
    -------
    dict[str, float]
        Statement to likelihood; the axis pseudo-statement has likelihood 0 (no information).
    """
    return {**scp_codes, **({f"axis:{heart_axis}": 0.0} if heart_axis else {})}


def aha_match(spec: str, statements: tuple[tuple[str, frozenset[str]], ...]) -> bool:
    """
    Whether an SPH record matches one AHA specification.

    Parameters
    ----------
    spec : str
        ``"B"``, ``"B+M"`` or ``"B="``.
    statements : tuple[tuple[str, frozenset[str]], ...]
        Output of ``parse_aha``.

    Returns
    -------
    bool
        True when a statement matches.
    """
    if "+" in spec:
        base, modifier = spec.split("+")
        return any(code == base and modifier in modifiers for code, modifiers in statements)
    if spec.endswith("="):
        return any(code == spec[:-1] and not modifiers for code, modifiers in statements)
    return any(code == spec for code, _ in statements)


def listed(specs: frozenset[str], record: object, vocabulary: str) -> bool:
    """
    Whether a record lists any of the given codes.

    Parameters
    ----------
    specs : frozenset[str]
        Code specifications of one vocabulary.
    record : object
        The record's codes: a dict of SCP statements (``ptbxl_codes``), a frozenset of SNOMED codes
        (``parse_snomed``) or SPH statements (``parse_aha``).
    vocabulary : str
        ``scp``, ``snomed`` or ``aha``.

    Returns
    -------
    bool
        True when one specification matches.
    """
    if vocabulary == "aha":
        return any(aha_match(spec, record) for spec in specs)
    return any(spec in record for spec in specs)


def annotated(key: str, source: str) -> bool:
    """
    Whether a source annotates a condition, so that its negatives count.

    Parameters
    ----------
    key : str
        Condition name in ``CONDITIONS``.
    source : str
        One of ``SOURCES``.

    Returns
    -------
    bool
        False when the vocabulary cannot code the condition, the source is listed in ``NOT_ANNOTATED``, or
        the source is CPSC 2018 and none of the condition's codes is one of its nine classes.
    """
    positive = CONDITIONS[key]["positive"][VOCABULARY[source]]
    cpsc_outside = source == "cpsc_2018" and not positive & CPSC_2018_CLASSES
    return bool(positive) and source not in NOT_ANNOTATED.get(key, frozenset()) and not cpsc_outside


def condition_state(key: str, source: str, record: object) -> float:
    """
    State of one condition on one record, before the contradictory-pair rule.

    Parameters
    ----------
    key : str
        Condition name in ``CONDITIONS``.
    source : str
        One of ``SOURCES``.
    record : object
        The record's codes in its source's vocabulary (see ``listed``).

    Returns
    -------
    float
        1.0, 0.0 or NaN.
    """
    vocabulary = VOCABULARY[source]
    definition = CONDITIONS[key]
    state = np.nan
    if annotated(key, source):
        if listed(definition["positive"][vocabulary], record, vocabulary):
            state = 1.0
        elif not listed(definition["ambiguous"][vocabulary], record, vocabulary):
            state = 0.0
    return state


def condition_states(source: str, record: object) -> dict[str, float]:
    """
    States of every condition in ``CONDITIONS`` on one record.

    Parameters
    ----------
    source : str
        One of ``SOURCES``.
    record : object
        The record's codes in its source's vocabulary (see ``listed``).

    Returns
    -------
    dict[str, float]
        1.0, 0.0 or NaN per condition. Both conditions of a contradictory pair are NaN when the record lists
        a positive code of each, whether or not the source annotates them.
    """
    vocabulary = VOCABULARY[source]
    states = {key: condition_state(key, source, record) for key in CONDITIONS}
    for pair in CONTRADICTORY_PAIRS:
        if all(listed(CONDITIONS[key]["positive"][vocabulary], record, vocabulary) for key in pair):
            states[pair[0]] = states[pair[1]] = np.nan
    return states


def low_likelihood_only(key: str, record: dict[str, float], diagnostic: frozenset[str]) -> bool:
    """
    Whether a PTB-XL positive rests only on diagnostic statements with likelihood below 50.

    Parameters
    ----------
    key : str
        Condition name in ``CONDITIONS``.
    record : dict[str, float]
        Output of ``ptbxl_codes``.
    diagnostic : frozenset[str]
        SCP statements flagged diagnostic in ``scp_statements.csv``. Form and rhythm statements carry no
        likelihood, so they never count as low likelihood.

    Returns
    -------
    bool
        True when the record lists positive statements of the condition and each one is diagnostic with a
        likelihood below ``LIKELIHOOD_THRESHOLD``.
    """
    hits = [(code, value) for code, value in record.items() if code in CONDITIONS[key]["positive"]["scp"]]
    return bool(hits) and all(code in diagnostic and value < LIKELIHOOD_THRESHOLD for code, value in hits)


def group_flags(source: str, record: object, classes: dict[str, str]) -> dict[str, float]:
    """
    Space-A group membership with the existing definitions, before undefined rules.

    Parameters
    ----------
    source : str
        One of ``SOURCES``.
    record : object
        The record's codes in its source's vocabulary (see ``listed``).
    classes : dict[str, str]
        PTB-XL diagnostic statement to superclass (``ecg_experiment.eda.ptbxl.diagnostic_classes``).

    Returns
    -------
    dict[str, float]
        1.0 or 0.0 per group of ``SPACE_A``.
    """
    vocabulary = VOCABULARY[source]
    if vocabulary == "scp":
        found = {classes[code] for code in record if code in classes}
        return {name: float(name in found) for name in SPACE_A}
    if vocabulary == "aha":
        bases = {base for base, _ in record}
        return {name: float(bool(bases & members)) for name, members in SPACE_A_CODES["aha"].items()}
    return {name: float(bool(record & members)) for name, members in SPACE_A_CODES["snomed"].items()}


def space_a_states(source: str, record: object, classes: dict[str, str]) -> dict[str, float]:
    """
    States of the space-A groups MI, STTC, CD and HYP on one record.

    The existing definitions (PTB-XL diagnostic superclasses at any likelihood, the Challenge and SPH
    superclass code sets) are kept, with SPH code 153 moved from STTC to HYP. A group is undefined at CPSC
    2018 when none of its codes is one of the nine classes (MI and HYP), and HYP is undefined when the only
    hypertrophy evidence is left ventricular high voltage.

    Parameters
    ----------
    source : str
        One of ``SOURCES``.
    record : object
        The record's codes in its source's vocabulary (see ``listed``).
    classes : dict[str, str]
        PTB-XL diagnostic statement to superclass.

    Returns
    -------
    dict[str, float]
        1.0, 0.0 or NaN per group.
    """
    vocabulary = VOCABULARY[source]
    states = group_flags(source, record, classes)
    if states["HYP"] == 0.0 and listed(LV_HIGH_VOLTAGE[vocabulary], record, vocabulary):
        states["HYP"] = np.nan
    if source == "cpsc_2018":
        for name, members in SPACE_A_CODES["snomed"].items():
            if not members & CPSC_2018_CLASSES:
                states[name] = np.nan
    return states


def normal_state(source: str, record: object) -> float:
    """
    Apply the source's own normal definition, kept for audits; it is not a class.

    Parameters
    ----------
    source : str
        One of ``SOURCES``.
    record : object
        The record's codes in its source's vocabulary (see ``listed``).

    Returns
    -------
    float
        1.0 for PTB-XL NORM with sinus rhythm as the only other statement, Challenge sinus rhythm as the only
        code, or SPH code 1 alone; 0.0 otherwise; NaN for CPSC-Extra, which holds no normal record.
    """
    vocabulary = VOCABULARY[source]
    if source in NORMAL_UNDEFINED:
        return np.nan
    if vocabulary == "scp":
        statements = {code for code in record if not code.startswith("axis:")}
        normal = "NORM" in statements and statements <= {"NORM", "SR"}
    elif vocabulary == "aha":
        normal = {base for base, _ in record} == {sph.NORMAL}
    else:
        normal = record == {challenge_labels.SINUS_RHYTHM}
    return float(normal)


def record_labels(source: str, record: object, classes: dict[str, str],
                  diagnostic: frozenset[str]) -> dict[str, float | bool]:
    """
    Every label column of one record.

    Parameters
    ----------
    source : str
        One of ``SOURCES``.
    record : object
        The record's codes in its source's vocabulary (see ``listed``).
    classes : dict[str, str]
        PTB-XL diagnostic statement to superclass.
    diagnostic : frozenset[str]
        PTB-XL diagnostic statements.

    Returns
    -------
    dict[str, float | bool]
        ``b_<condition>`` for space B, ``c_<condition>`` for space C, ``a_<group>`` for space A, ``normal``,
        and ``lik50_<condition>`` for space B: True when a PTB-XL positive rests only on diagnostic
        statements with likelihood below 50 (undefined in the sensitivity analysis).
    """
    states = condition_states(source, record)
    labels: dict[str, float | bool] = {f"b_{key}": states[key] for key in SPACE_B}
    labels |= {f"c_{key}": states[key] for key in SPACE_C}
    labels |= {f"a_{name}": value for name, value in space_a_states(source, record, classes).items()}
    labels["normal"] = normal_state(source, record)
    for key in SPACE_B:
        flag = source == "ptbxl" and states[key] == 1.0 and low_likelihood_only(key, record, diagnostic)
        labels[f"lik50_{key}"] = bool(flag)
    return labels


def sensitivity_state(state: float, low_likelihood: bool) -> float:
    """
    State of one condition in the likelihood sensitivity analysis.

    Parameters
    ----------
    state : float
        Primary state: 1.0, 0.0 or NaN.
    low_likelihood : bool
        The ``lik50_<condition>`` flag.

    Returns
    -------
    float
        NaN when the flag is set, the primary state otherwise.
    """
    return np.nan if low_likelihood else state


def state_name(state: float) -> str:
    """
    Readable name of a label state.

    Parameters
    ----------
    state : float
        1.0, 0.0 or NaN.

    Returns
    -------
    str
        ``positive``, ``negative`` or ``undefined``.
    """
    return "undefined" if math.isnan(state) else STATES[state]


def mapping_rows() -> list[dict[str, str]]:
    """
    One row per condition, vocabulary and code specification.

    Returns
    -------
    list[dict[str, str]]
        ``condition``, ``name``, ``vocabulary``, ``code`` and ``role`` (``positive`` or ``ambiguous``);
        a vocabulary that cannot code the condition gets one row with an empty code and role
        ``not codable``.
    """
    rows = []
    for key, definition in CONDITIONS.items():
        for vocabulary in ("scp", "snomed", "aha"):
            specs = [(code, role) for role in ("positive", "ambiguous") for code in
                     sorted(definition[role][vocabulary])]
            for code, role in specs or [("", "not codable")]:
                rows.append({"condition": key, "name": definition["name"], "vocabulary": vocabulary,
                             "code": code, "role": role})
    return rows


def status_rows() -> list[dict[str, str]]:
    """
    Annotation status of every condition and space-A group in every source.

    Returns
    -------
    list[dict[str, str]]
        ``label`` (``b_``, ``c_`` or ``a_`` column name), ``source`` and ``status``: ``annotated``,
        ``partial`` (labels are still defined; see ``PARTIAL``), ``not annotated`` or ``not codable``.
    """
    rows = []
    keys = [f"b_{key}" for key in SPACE_B] + [f"c_{key}" for key in SPACE_C if key not in SPACE_B]
    for column in keys:
        key = column[2:]
        for source in SOURCES:
            status = "annotated" if annotated(key, source) else "not annotated"
            if not CONDITIONS[key]["positive"][VOCABULARY[source]]:
                status = "not codable"
            elif (source, key) in PARTIAL:
                status = "partial"
            rows.append({"label": column, "source": source, "status": status})
    for name in SPACE_A:
        for source in SOURCES:
            outside = source == "cpsc_2018" and not SPACE_A_CODES["snomed"][name] & CPSC_2018_CLASSES
            status = "not annotated" if outside else "partial" if (source, name) in PARTIAL else "annotated"
            rows.append({"label": f"a_{name}", "source": source, "status": status})
    return rows
