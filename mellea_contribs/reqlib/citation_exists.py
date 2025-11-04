from mellea.stdlib.requirement import Requirement, ValidationResult
from mellea.stdlib.base import Context, CBlock

import json
import os
import re
from eyecite import get_citations, clean_text
from typing import Any

# region: citation_exists function and helpers

def normalize_case_name(name) -> str:
    """
    Converts a case name to a standard format.

    Args:
        name: A string representing the case name.

    Returns:
        A normalized case name.
    """
    # 1. Lowercase everything
    name = name.lower()

    # 2. Normalize 'vs', 'vs.', 'v', 'versus' to 'v.'
    name = re.sub(r'\b(vs\.?|versus|v)(?!\.)\b', 'v.', name)

    # 3. Remove all non-alphanumeric characters except periods, spaces, and apostrophes
    name = re.sub(r"[^a-z0-9.& ']+", '', name)

    # 4. Replace multiple spaces with a single space
    name = re.sub(r'\s+', ' ', name)

    return name.strip()

# might not be needed
# def ensure_list_of_dicts(obj: Any) -> list[dict]:
#     """
#     Normalize any JSON-like object into a list of dictionaries.

#     Accepts:
#       - A JSON string (object or array)
#       - A single dict
#       - A list of dicts

#     Args:
#         obj: Any data type, ideally something that can unpacked into a dictionary

#     Returns:
#         The unpacked object in list of dictionary form or raises an error.
#     """
#     # JSON string
#     if isinstance(obj, str):
#         try:
#             obj = json.loads(obj)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Invalid JSON string: {e!s}")

#     # Single dict
#     if isinstance(obj, dict):
#         return [obj]

#     # List of dicts
#     if isinstance(obj, list):
#         if all(isinstance(item, dict) for item in obj):
#             return obj
#         else:
#             raise ValueError("List contains non-dictionary elements")

#     raise TypeError(f"Unsupported metadata format: {type(obj)}")

# alternatively:
# should this take in last_output instead of the whole context?
# get case name: take LLM output and extract case name --> a string which you get from ctx.last_output() is the input
# so the argument should be ctx.last_output.value: str

def extract_case_names(ctx: Context) -> list[str]:
    """
    Given an LLM output, use eyecite to parse the text and collect case names.

    Args:
        ctx: An LLM output that may contain multiple citations.

    Returns:
        A list of case names.
    """
    # should i clean text??

    # install hyperscan if not already installed
    # !pip install hyperscan
    # tokenizer = HyperscanTokenizer(cache_dir=".test_cache")
    # citations = get_citations(cleaned_text, tokenizer=tokenizer)

    # or this?
    # cleaned_text = clean_text(text, ["html", "all_whitespace"])
    # citations = get_citations(cleaned_text)

    # get_citations outputs a list of citations
    citations = get_citations(ctx.last_output().value)
    case_names = set()

    for citation in citations:
        plaintiff = citation.metadata.get("plaintiff")
        defendant = citation.metadata.get("defendant")
        if plaintiff and defendant:
            case_names.add(f"{plaintiff} v. {defendant}")
            # name = citation.metadata['plaintiff'] + " v. " + citation.metadata['defendant']
            # case_names.add(name)
        
    return list(case_names)

def citation_exists(ctx: Context, case_metadata: list[dict]) -> ValidationResult:
    """
    Given an LLM output and a list of dictionaries, checks that list (which represents a collection of
    case metadata json files) to see if the given case names can be found in it.

    Args:
        ctx: Context that contains the case names we're checking for
        case_metadata: a list of dictionaries which represents a collection of case metadata json files

    Returns:
        A validation result indicating if a match was found between given case names and database
    """
    if ctx is None:
        return ValidationResult(False, reason="No context provided in output")
    
    # 1) this will spit out a bunch of words --> look through to extract case names
    # 2) use eyecite (might have to do some conversion)
    last_output = ctx.last_output() 

    # if last_output is None or not getattr(output, "value", None):
    if last_output is None:
        return ValidationResult(False, reason="No last output found in context")
    
    # 3) run checking
    # call get_case_name func
    case_names = extract_case_names(ctx)

    if not case_names or not isinstance(case_names, list[str]):
        return ValidationResult(False, reason="No case names provided in output")
    
    normalized_case_names = [normalize_case_name(case_name) for case_name in case_names]
    
    case_names = set()
    case_name_abb = set()

    # add name and name_abbreviation from the database
    for case in case_metadata:
        if 'name' in case:
            case_names.add(normalize_case_name(case['name']))
        if 'name_abbreviation' in case:
            case_name_abb.add(normalize_case_name(case['name_abbreviation']))

    # Check both name and name_abbreviation
    for normalized_case_name in normalized_case_names:
        if normalized_case_name not in case_names and normalized_case_name not in case_name_abb:
            # probably want to change this to the actual case name at some point
            # maybe keep a tuple structure or something
            return ValidationResult(False, reason=f"'{normalized_case_name}' not found in database")
        
    return ValidationResult(True, reason="All case names found in database")

    # check if this code chunk is right later
    # db_names = {normalize_case_name(c["name"]) for c in case_metadata if "name" in c}
    # db_abbrevs = {
    #     normalize_case_name(c["name_abbreviation"]) for c in case_metadata if "name_abbreviation" in c
    # }

    # for name in normalized_output_names:
    #     if name not in db_names and name not in db_abbrevs:
    #         return ValidationResult(False, reason=f"Case '{name}' not found in database")

    # return ValidationResult(True, reason="All case names found in database")


class CaseNameExistsInDatabase(Requirement):
    """
    Checks if the output case name exists in the provided case metadata database.
    """
    def __init__(self, case_metadata: str):
        self._case_metadata = case_metadata
        super().__init__(
            description="The case name should exist in the provided case metadata database.",
            validation_fn=lambda ctx: citation_exists(ctx, self._case_metadata),
        )
# endregion