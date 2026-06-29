"""Plate text normalization, country parsing, and Dutch sidecode checks."""

import re

# Official vehicle-registration country codes that appear on European plates
# (EU blue-band codes + wider EEA / neighbouring-state codes).
PLATE_COUNTRY_CODES = {
    # EU member states
    "A": "Austria",
    "B": "Belgium",
    "BG": "Bulgaria",
    "CY": "Cyprus",
    "CZ": "Czech Republic",
    "D": "Germany",
    "DK": "Denmark",
    "E": "Spain",
    "EST": "Estonia",
    "F": "France",
    "FIN": "Finland",
    "GR": "Greece",
    "H": "Hungary",
    "HR": "Croatia",
    "I": "Italy",
    "IRL": "Ireland",
    "L": "Luxembourg",
    "LT": "Lithuania",
    "LV": "Latvia",
    "M": "Malta",
    "NL": "Netherlands",
    "P": "Portugal",
    "PL": "Poland",
    "RO": "Romania",
    "S": "Sweden",
    "SK": "Slovakia",
    "SLO": "Slovenia",
    # EEA / candidate / neighbouring states
    "AL": "Albania",
    "AND": "Andorra",
    "ARM": "Armenia",
    "AZ": "Azerbaijan",
    "BA": "Bosnia & Herzegovina",
    "BY": "Belarus",
    "CH": "Switzerland",
    "FL": "Liechtenstein",
    "GB": "United Kingdom",
    "GE": "Georgia",
    "IS": "Iceland",
    "KS": "Kosovo",
    "MD": "Moldova",
    "ME": "Montenegro",
    "MK": "North Macedonia",
    "N": "Norway",
    "RS": "Serbia",
    "RUS": "Russia",
    "TR": "Turkey",
    "UA": "Ukraine",
    "UK": "United Kingdom",
}

# Pattern: optional country prefix (1-3 letters) then separator then remainder
_COUNTRY_RE = re.compile(r"^([A-Z]{1,3})[-\s](.+)$")

# X = letter (A-Z), 9 = digit (0-9).
# Each entry: (sidecode_number, display_pattern, regex_with_dashes, regex_no_sep)
_DUTCH_SIDECODES = [
    (1, "XX-99-99", re.compile(r"^[A-Z]{2}-\d{2}-\d{2}$"), re.compile(r"^[A-Z]{2}\d{4}$")),
    (2, "99-99-XX", re.compile(r"^\d{2}-\d{2}-[A-Z]{2}$"), re.compile(r"^\d{4}[A-Z]{2}$")),
    (3, "99-XX-99", re.compile(r"^\d{2}-[A-Z]{2}-\d{2}$"), re.compile(r"^\d{2}[A-Z]{2}\d{2}$")),
    (4, "XX-99-XX", re.compile(r"^[A-Z]{2}-\d{2}-[A-Z]{2}$"), re.compile(r"^[A-Z]{2}\d{2}[A-Z]{2}$")),
    (5, "XX-XX-99", re.compile(r"^[A-Z]{2}-[A-Z]{2}-\d{2}$"), re.compile(r"^[A-Z]{4}\d{2}$")),
    (6, "99-XX-XX", re.compile(r"^\d{2}-[A-Z]{2}-[A-Z]{2}$"), re.compile(r"^\d{2}[A-Z]{4}$")),
    (7, "99-XXX-9", re.compile(r"^\d{2}-[A-Z]{3}-\d$"), re.compile(r"^\d{2}[A-Z]{3}\d$")),
    (8, "9-XXX-99", re.compile(r"^\d-[A-Z]{3}-\d{2}$"), re.compile(r"^\d[A-Z]{3}\d{2}$")),
    (9, "XX-999-X", re.compile(r"^[A-Z]{2}-\d{3}-[A-Z]$"), re.compile(r"^[A-Z]{2}\d{3}[A-Z]$")),
    (10, "X-999-XX", re.compile(r"^[A-Z]-\d{3}-[A-Z]{2}$"), re.compile(r"^[A-Z]\d{3}[A-Z]{2}$")),
    (11, "XXX-99-X", re.compile(r"^[A-Z]{3}-\d{2}-[A-Z]$"), re.compile(r"^[A-Z]{3}\d{2}[A-Z]$")),
]


def normalise_plate(text: str) -> str:
    """Upper-case, strip noise, collapse spaces."""
    t = text.upper().strip()
    t = re.sub(r"[^A-Z0-9 \-]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_country(text: str):
    """
    Returns (country_code, country_name, plate_number).
    country_code and country_name are None when no country is detected.
    """
    m = _COUNTRY_RE.match(text)
    if m:
        prefix = m.group(1)
        if prefix in PLATE_COUNTRY_CODES:
            return prefix, PLATE_COUNTRY_CODES[prefix], m.group(2).strip()

    for code in sorted((c for c in PLATE_COUNTRY_CODES if len(c) >= 2), key=len, reverse=True):
        if text.startswith(code):
            remainder = text[len(code) :]
            if len(remainder) >= 4 and remainder[0].isalnum():
                return code, PLATE_COUNTRY_CODES[code], remainder.strip()

    if text in PLATE_COUNTRY_CODES:
        return text, PLATE_COUNTRY_CODES[text], ""

    return None, None, text


def match_dutch_sidecode(text: str):
    """
    Check whether text matches a Dutch licence plate sidecode pattern.
    Returns (sidecode_number, pattern_string) or (None, None).
    """
    t = text.upper().strip()
    for sc, pat, re_dash, _ in _DUTCH_SIDECODES:
        if re_dash.match(t):
            return sc, pat

    t_plain = re.sub(r"[-\s]", "", t)
    for sc, pat, _, re_plain in _DUTCH_SIDECODES:
        if re_plain.match(t_plain):
            return sc, pat

    return None, None
