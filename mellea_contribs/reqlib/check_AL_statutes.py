"""Alabama statute citation validation."""

import re

from mellea.stdlib.base import Context
from mellea.stdlib.requirement import Requirement, ValidationResult

from .statute_data import alabama


def parse_AL(file: str) -> list[str]:
    """Parse Alabama statute citations from the provided file content."""
    citations = []
    pattern = r"Ala\. Code §"
    matches = [m.start() for m in re.finditer(pattern, file)]
    for match in matches:
        end = re.search(r"\d{4}\)", file[match + 1 :])
        if end is not None:
            citations.append(file[match : end.end() + match + 1])
        else:
            raise Exception(
                f"Could not find closing parenthesis for statute match: {file[match:]}"
            )
    return citations


def check_AL(citations: list[str]) -> list[bool]:
    """Check the existence of Alabama statutes from the provided citations."""
    statute_exists = []
    for citation in citations:
        section_symbol = citation.find("§")
        start = re.search(r"[1-9]", citation[section_symbol:]).start() + section_symbol
        end = citation[start:].find(" ")
        statute = citation[start : start + end]
        title, section, rest = None, None, None
        try:
            title, section, rest = statute.split("-")
        except:
            # Parsed incorrectly
            statute_exists.append(False)
            continue
        if rest is None:
            statute_exists.append(False)
            continue
        if title not in alabama:
            statute_exists.append(False)
            continue
        if section not in alabama[title]:
            statute_exists.append(False)
            continue
        search = alabama[title][section]
        if float(rest) in search:
            statute_exists.append(True)
            continue
        if isinstance(search, list):
            found = False
            for item in search:
                if isinstance(item, tuple):
                    if int(rest) >= item[0] and int(rest) < item[1]:
                        statute_exists.append(True)
                        found = True
                        break
                else:
                    continue
            if not found:
                statute_exists.append(False)
                continue
        else:
            if "." not in rest:
                statute_exists.append(False)
                continue
            [a, b] = rest.split(".")
            if a in search:
                found = False
                for item in search[a]:
                    if isinstance(item, tuple):
                        if int(b) >= item[0] and int(b) < item[1]:
                            statute_exists.append(True)
                            found = True
                            break
                    else:
                        continue
                if not found:
                    statute_exists.append(False)
                    continue
            else:
                statute_exists.append(False)
                continue
    return statute_exists


def get_AL_statutes(ctx: Context) -> list[str]:
    """Extract Alabama statute citations from the provided file content."""
    if ctx is None:
        raise ValueError("Context is required to extract Alabama statutes.")

    last_output = ctx.last_output()
    if last_output is None:
        raise ValueError("No text found in the last output of the context.")
    text = last_output.value
    if not text or not isinstance(text, str):
        raise ValueError(
            "The last output must be a string containing the file content."
        )
    return parse_AL(text)


def validate_AL_statutes(citations: list[str]) -> ValidationResult:
    """Validate the existence of cited Alabama statutes."""
    results = check_AL(citations)
    all_exist = True
    for exists in results:
        if not exists:
            all_exist = False
            break
    if all_exist:
        return ValidationResult(True)
    else:
        return ValidationResult(
            False,
            reason=f"These statutes do not exist: {[citations[i] for i, exists in enumerate(results) if not exists]}",
        )


class VerifyALStatutes(Requirement):
    def __init__(self):
        super().__init__(
            description="Verify the existence of Alabama statutes in the provided citations.",
            validation_fn=lambda ctx: validate_AL_statutes(get_AL_statutes(ctx)),
        )
