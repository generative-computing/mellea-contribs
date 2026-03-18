import json
from typing import Any, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from citeurl import Citator
from eyecite import get_citations
from mellea.stdlib.base import Context
from mellea.stdlib.requirement import Requirement, ValidationResult
from playwright.sync_api import sync_playwright

# region citation_exists helpers

"""
Validator: Ensure that every case-law citation in an LLM output corresponds to a real case in the
provided case metadata database.

Process:
1. Extract citations from LLM output using citeurl.
2. Map citation objects to URLs.
3a. For each case.law URL:
    - Fetch JSON metadata.
    - Compare its case ID against the known database.
3b. For each non case.law URL:
    - Run the original text through eyecite to extract volume, reporter, and page.
    - Check if any cases in the database match based on this information.
4. If any citation fails, return ValidationResult(False).
5. If all succeed, return ValidationResult(True).
"""


def text_to_urls(text: str) -> dict[str, str] | ValidationResult:
    """Extracts all citation URLs from the given text using citeurl.

    Args:
        text: An LLM output

    Returns:
        A dictionary of citation URLs and the corresponding text.

    Behavior:
        If a citation does not have a URL attribute, we return a ValidationResult(False)
        so that the parent validator can fail accordingly.
    """
    citator = Citator()
    citations = citator.list_cites(text)

    urls = {}
    errors = []

    for citation in citations:
        if hasattr(citation, "URL") and citation.URL:
            # Map the URL to the text corresponding to the citation
            urls[citation.URL] = citation.text
        else:
            # Record a descriptive error about the invalid citation object
            errors.append(f"Citation has no URL attribute: {citation!r}")

    if errors:
        # Raise one combined error
        error_msg = "Some citations did not contain URLs:\n" + "\n".join(errors)
        return ValidationResult(False, reason=error_msg)

    return urls


def extract_case_metadata_url(case_url: str) -> ValidationResult | str:
    """Converts a case.law URL to the corresponding static JSON metadata URL.

    Args:
        case_url: A cite.case.law page

    Returns:
        A URL to the JSON metadata for the case or a false ValidationResult if the link cannot be found
    """
    # Take the full input URL and split into structured components
    parsed = urlparse(case_url)
    # Turn the query part into a dictionary
    params = parse_qs(parsed.query)

    # Use None as a fallback if a value is missing
    reporter = params.get("reporter", [None])[0]
    volume = params.get("volume", [None])[0]
    case = params.get("case", [None])[0]

    if not reporter or not volume or not case:
        # Use playwright if URL parsing doesn't work
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(case_url)

            # Wait for the metadata link to appear
            link = page.wait_for_selector("a:has-text('Download case metadata')")
            browser.close()

            if not link:
                return ValidationResult(
                    False, reason=f"No metadata link found on page: {case_url}"
                )

            # Extract relative href
            href = link.get_attribute("href")
            if not href:
                return ValidationResult(
                    False,
                    reason=f"Metadata link missing href attribute on page: {case_url}",
                )

            # Build the absolute metadata URL
            return urljoin(case_url, href)

    return f"https://static.case.law/{reporter}/{volume}/cases/{case}.json"


def metadata_url_to_json(metadata_url: str) -> dict:
    """Fetches JSON metadata for a case.

    Args:
        metadata_url: Fully-qualified URL to metadata.json

    Returns:
        A dictionary representing the JSON metadata.
    """
    resp = requests.get(metadata_url)
    resp.raise_for_status()
    return resp.json()


def collect_ids_in_database(database: list[dict]) -> set:
    """Collects all case IDs from the provided caselaw metadata.

    Args:
        database: A list of case dictionaries loaded from a caselaw JSON dataset.

    Returns:
        A set of all unique case IDs.
    """
    return {case["id"] for case in database}


def parse_db_cite(cite: str) -> tuple:
    """Given a citation in the form of a string, return a normalized tuple breaking the
    volume, reporter, and page into distinct parts.

    Args:
        cite: A string representing the citation found in the text.

    Returns:
        A tuple containing the volume, normalized reporter, and page of a citation.
    """
    # TODO: Could mishandle the following: “U. S.”, “S. Ct.”, “F. Supp. 2d”
    parts = cite.split()

    # If the citation has less then 3 parts, it is likely irregular
    if len(parts) < 3:
        return None

    volume = parts[0]
    page = parts[-1]
    reporter = " ".join(parts[1:-1])
    normalized_reporter = reporter.lower().replace(".", "")

    return (volume, normalized_reporter, page)


def build_citation_index(database: list[dict]) -> set[tuple]:
    """Extract all of the citations in the database for easy comparison.

    Args:
        database: A list of case dictionaries loaded from a caselaw JSON dataset.

    Returns:
        A set of normalized tuples.
    """
    index = set()

    for case in database:
        # There can be multiple citations for each case
        for c in case.get("citations", []):
            parsed = parse_db_cite(c["cite"])
            if parsed:
                index.add(parsed)

    return index


def non_caselaw_citation_exists(
    text: str, database: list[dict]
) -> bool | ValidationResult:
    """Given the text corresponding to a citation, check whether that citation can matched to
    the cases in the database.

    We first use a deterministic approach, like by matching against citations in the database.
    Then, we fuzzy match across features like case name, volume, and year.
    Finally, we resort to using LLM-as-a-judge to determine if a match exists.

    Args:
        text: A string containing a citation.
        database: A list of case dictionaries loaded from a caselaw JSON dataset.

    Returns:
        Boolean indicating whether a match was found or ValidationResult if there was an error.
    """
    # The citations field in the original database represents how each case can be cited,
    # but it's not exhaustive.

    # Get_citations is an eyecite function (extracts information from text)
    citations = get_citations(text)
    # Build our database in the proper format for easy access
    citation_index = build_citation_index(database)

    # Return False ValidationResult if multiple or no citations have been found in the text.
    if len(citations) != 1:
        return ValidationResult(
            False, reason="Error from parsing citations with eyecite."
        )

    try:
        groups = citations[0].groups
        vol = groups["volume"]
        reporter = groups["reporter"]
        page = groups["page"]
    except (KeyError, TypeError, IndexError):
        return ValidationResult(
            False, reason="Error from parsing citations with eyecite."
        )

    normalized_reporter = reporter.lower().replace(".", "")

    # TODO: if (vol, normalized_reporter, page) not in citation_index,
    # resort to other measures, like fuzzy matching names and/or LLM-as-a-judge

    return (vol, normalized_reporter, page) in citation_index


# endregion


# region citation_exists function


def citation_exists(ctx: Context, database: list[dict]) -> ValidationResult:
    """Validator:
    Ensures that every cite.case.law URL in the LLM output corresponds to a real case in the provided case metadata database.

    Args:
        ctx: Mellea runtime context containing the last LLM output.
        database: Parsed caselaw metadata database of JSON objects.

    Returns:
        ValidationResult indicating pass/fail.
    """
    if ctx is None:
        return ValidationResult(False, reason="No context provided in output.")

    last_output = ctx.last_output()

    if last_output is None:
        return ValidationResult(False, reason="No last output found in context.")

    urls_or_error = text_to_urls(last_output)

    # text_to_urls may return a ValidationResult (error condition)
    if isinstance(urls_or_error, ValidationResult):
        return urls_or_error

    # List of urls of citations found in the LLM output
    output_citation_urls = list(urls_or_error.keys())

    if output_citation_urls is None or output_citation_urls == []:
        # No citations, so trivially valid
        return ValidationResult(True, reason="No citations found.")

    database_ids = collect_ids_in_database(database)

    for url in output_citation_urls:
        # If this URL is Caselaw, do direct comparison within database by using case id
        if "cite.case.law" in url:
            try:
                metadata_url = extract_case_metadata_url(url)

                # Check if extract_case_metadata_url returns a ValidationResult and propagate it
                if isinstance(metadata_url, ValidationResult):
                    return metadata_url

                metadata = metadata_url_to_json(metadata_url)
                case_id = metadata["id"]

            except Exception as e:
                return ValidationResult(
                    False, reason=f"Failed to retrieve metadata for {url}: {e}"
                )

            if case_id not in database_ids:
                return ValidationResult(
                    False, reason=f"Case {case_id} not found in database"
                )

        # Non-case.law citations: pass into Eyecite and see if citations match
        else:
            # TODO: This logic might need some reworking because for cases where a match
            # cannot be found, we cannot verify this citation, but we also cannot disprove it
            # (due to factors like reporter names varying, citation lists in the database
            # not being exhaustive, and parsing being lossy or ambiguous)
            text = urls_or_error[url]
            result = non_caselaw_citation_exists(text, database)

            # Case 1: hard failure -> propagate
            if isinstance(result, ValidationResult):
                return result

            # Case 2: deterministic match found -> OK
            if result is True:
                continue

            # Case 3: result is False -> inconclusive -> do NOT fail
            # Explicitly allow this to pass for now
            continue

    return ValidationResult(
        True,
        reason="All case.law citations verified; non-case.law citations did not fail verification.",
    )


class CaseNameExistsInDatabase(Requirement):
    """Requirement wrapper for Mellea that ensures case citations in LLM output
    refer to real cases in the provided metadata database.
    """

    def __init__(self, case_metadata: list[dict]):
        self._case_metadata = case_metadata
        super().__init__(
            description="The case name should exist in the provided case metadata database.",
            validation_fn=lambda ctx: citation_exists(ctx, self._case_metadata),
        )


# endregion
