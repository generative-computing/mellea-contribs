import json
import os
import re

from mellea.stdlib.base import Context
from mellea.stdlib.requirement import Requirement, ValidationResult

# ------------------------------------------------------------ #
# FUNCTIONS SUPPORTING CASE NAME AS INPUT TO IS_APPELLATE_CASE #
# ------------------------------------------------------------ #


def load_jsons_from_folder(folder_path: str) -> list[dict]:
    """Load all JSON files in the folder into a list of dicts."""
    all_entries = []
    for file_name in os.listdir(folder_path):
        with open(os.path.join(folder_path, file_name)) as file:
            data = json.load(file)
            all_entries.extend(data)
    return all_entries


def get_court_from_case(case_name: str, case_metadata: list[dict]) -> str:
    """Given a case name and case metadata, return the court name."""
    for entry in case_metadata:
        if case_name.lower() in entry["name"].lower():
            return entry["court"]["name"]
    raise ValueError("Court not found for the given case name")


def is_appellate_court_fullname(court_name: str) -> ValidationResult:
    """Determine if a court is an appellate court based on its full name."""
    # rule exceptions: the 2 appellate courts whose names do not include the below keywords
    rule_exceptions = ["pennsylvania superior court", "pennsylvania commonwealth court"]
    keywords = ["supreme", "appeal", "appellate"]

    lowered_name = court_name.lower()
    return ValidationResult(
        any(keyword in lowered_name for keyword in keywords)
        or lowered_name in rule_exceptions
    )


# ----------------------------------------------------------- #
# FUNCTIONS SUPPORTING CITATION AS INPUT TO IS_APPELLATE_CASE #
# ----------------------------------------------------------- #


def court_abbv_from_citation(citation: str) -> str:
    """Extract the court abbv from a legal case citation."""
    # case 1: if court abbv is in ending parentheses
    paren_pattern = r"\(([^)]*(?:Cir\.|D\.|S\.D\.|N\.D\.|E\.D\.|W\.D\.|M\.D\.|Ct\.|App\.|Sup\.).*?)\s*\d{4}\)"
    match = re.search(paren_pattern, citation)
    if match:
        court_text = match.group(1).strip()
        court_text = re.sub(
            r"\s*\d{4}$", "", court_text
        )  # remove year if it's at the end
        return court_text

    # case 2: reporter-based court identification; court abbv is between volume number and page number
    reporter_pattern = r"\d+\s+([A-Z][A-Za-z.&\d\s]+?)\s+\d+(?:\s+\(|$)"
    match = re.search(reporter_pattern, citation)
    if match:
        reporter = match.group(1).strip()
        if reporter in ["U.S.", "S.Ct.", "L.Ed.", "L.Ed.2d"]:
            return "U.S."
        return reporter

    raise ValueError(f"Could not extract court name from citation: {citation}")


def is_appellate_court_abbv(court_abbv: str) -> ValidationResult:
    """Determine if a court is an appellate court based on its abbreviated name."""
    court_lower = court_abbv.lower()

    # Supreme Courts (federal and state) -> appellate
    if any(
        indicator in court_lower for indicator in ["u.s.", "supreme", "sup.", "s.ct."]
    ):
        return ValidationResult(True)

    # Federal Circuit Courts -> appellate
    if "cir." in court_lower or re.search(r"\d+(st|nd|rd|th)\s+cir", court_lower):
        return ValidationResult(True)

    # Federal appellate reporters (F., F.2d, F.3d (word boundaries)) -> appellate
    if re.search(r"\bf\.\d*d\b", court_lower):
        return ValidationResult(True)

    # State supreme court reporters -> appellate
    state_supreme_abbrevs = {
        "ala.",
        "alaska",
        "ariz.",
        "ark.",
        "cal.",
        "colo.",
        "conn.",
        "del.",
        "d.c.",
        "fla.",
        "ga.",
        "haw.",
        "idaho",
        "ill.",
        "ind.",
        "iowa",
        "kan.",
        "ky.",
        "la.",
        "me.",
        "md.",
        "mass.",
        "mich.",
        "minn.",
        "miss.",
        "mo.",
        "mont.",
        "neb.",
        "nev.",
        "n.h.",
        "n.j.",
        "n.m.",
        "n.y.",
        "n.c.",
        "n.d.",
        "ohio",
        "okla.",
        "or.",
        "pa.",
        "r.i.",
        "s.c.",
        "s.d.",
        "tenn.",
        "tex.",
        "utah",
        "vt.",
        "va.",
        "wash.",
        "w. va.",
        "wis.",
        "wyo.",
    }
    if court_lower in state_supreme_abbrevs:
        return ValidationResult(True)

    # State appellate courts -> appellate
    if any(indicator in court_lower for indicator in ["app.", "court of appeals"]):
        return ValidationResult(True)

    # Caught all appellate cases, rest are not appellate
    return ValidationResult(False)


# ----------------- #
# IS_APPELLATE_CASE #
# ----------------- #


def is_appellate_case(ctx: Context, folder_path: str) -> ValidationResult:
    """Determine if the input (case name or citation) is an appellate court case.
    folder_path is needed in case the input is a case name.
    """
    # input validation
    if ctx is None:
        raise ValueError("Context cannot be None")

    last_output = ctx.last_output()
    if last_output is None:
        raise ValueError("No last output found in context")

    input = last_output.value

    if not input or not isinstance(input, str):
        raise ValueError("Input must be a non-empty string")

    # branch depending on input
    try:
        return is_appellate_court_abbv(court_abbv_from_citation(input))
    except ValueError:
        return is_appellate_court_fullname(
            get_court_from_case(input, load_jsons_from_folder(folder_path))
        )


class IsAppellateCase(Requirement):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        super().__init__(
            description="The result should be an appellate court case name or citation.",
            validation_fn=lambda ctx: is_appellate_case(ctx, folder_path),
        )
