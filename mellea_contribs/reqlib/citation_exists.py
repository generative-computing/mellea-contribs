from mellea.stdlib.requirement import Requirement, ValidationResult
from mellea.stdlib.base import Context, CBlock
from eyecite.models import FullCaseCitation, CitationBase
from eyecite import get_citations
from citeurl import Citator
from typing import Any, Optional
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin

import json
import os
import re
import requests

# region: citation_exists function and helpers

"""
Validator: Ensure that every case-law citation in an LLM output corresponds to a real case in the
provided case metadata database.

Process:
1. Extract citations from LLM output using citeurl.
2. Convert citation objects to URLs.
3. For each cite.case.law URL:
       - Use Playwright to extract metadata URL.
       - Fetch JSON metadata.
       - Compare its case ID against the known database.
4. If any citation fails, return ValidationResult(False).
5. If all succeed, return ValidationResult(True).
"""

def text_to_urls(text: str) -> list[str]:
    """
    Extracts all citation URLs from the given text using citeurl.

    Args:
        text: An LLM output

    Returns:
        A list of citation URLs.

    Behavior:
        - If a citation does not have a URL attribute, we return a ValidationResult(False)
          so that the parent validator can fail accordingly.
    """
    citator = Citator()
    citations = citator.list_cites(text)

    urls = []
    errors = []

    for citation in citations:
        if hasattr(citation, "URL") and citation.URL:
            urls.append(citation.URL)
        else:
            # Record a descriptive error about the invalid citation object
            errors.append(f"Citation has no URL attribute: {repr(citation)}")

    if errors:
        # Raise one combined error
        error_msg = "Some citations did not contain URLs:\n" + "\n".join(errors)
        return ValidationResult(False, reason=error_msg)

    return urls


def extract_case_metadata_url(page_url: str) -> str:
    """
    Visits a cite.case.law page using Playwright and extracts the "Download case metadata" link.

    Args:
        page_url: A cite.case.law page

    Returns:
        A URL to the JSON metadata for the case or a false ValidationResult if the link cannot be found
    """
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(page_url)
        
        # Wait for the metadata link to appear
        link = page.wait_for_selector("a:has-text('Download case metadata')")
        if not link:
            return ValidationResult(False, reason=f"No metadata link found on page: {page_url}")

        # Extract relative href
        href = link.get_attribute("href")
        if not href:
            return ValidationResult(False, reason=f"Metadata link missing href attribute on page: {page_url}")

        # Build the absolute metadata URL
        return urljoin(page_url, href)
    

def metadata_url_to_json(metadata_url: str) -> dict:
    """
    Fetches JSON metadata for a case.

    Args:
        metadata_url: Fully-qualified URL to metadata.json

    Returns:
        A dictionary representing the JSON metadata.
    """
    resp = requests.get(metadata_url)
    resp.raise_for_status()
    return resp.json()


def collect_ids_in_database(database: list[dict]) -> set:
    """
    Collects all case IDs from the provided caselaw metadata.

    Args:
        database: A list of case dictionaries loaded from a caselaw JSON dataset.

    Returns:
        A set of all unique case IDs.
    """
    return {case["id"] for case in database}


def citation_exists(ctx: Context, database: list[dict]) -> ValidationResult:
    """
    Validator:
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
        return ValidationResult(False, reason="No last output found in contex.")
    
    if type(last_output) != str:
        return ValidationResult(False, reason="Last output must be a string.")
    
    # List of urls of citations found in the LLM output
    output_citation_urls = text_to_urls(last_output)

    # text_to_urls may return a ValidationResult (error condition)
    if isinstance(output_citation_urls, ValidationResult):
        return output_citation_urls
    
    if output_citation_urls is None or output_citation_urls == []:
        # No citations, so trivially valid
        return ValidationResult(True, reason="No citations found.")

    database_ids = collect_ids_in_database(database)

    for url in output_citation_urls:

        # If this URL is Caselaw, do direct comparison within database by using case id
        if "cite.case.law" in url:
            try:
                metadata_url = extract_case_metadata_url(url)
                metadata = metadata_url_to_json(metadata_url)
                case_id = metadata["id"]

            except Exception as e:
                return ValidationResult(False, reason=f"Failed to retrieve metadata for {url}: {e}")
            
            if case_id not in database_ids:
                return ValidationResult(False, reason=f"Case {case_id} not found in database")
        
        else:
            # Non-caselaw citations (e.g., statutes): ignore
            # Extending functionality to be done later: use LLM as judge to see if citations match
            continue

    return ValidationResult(True, reason="All case names found in database")
    

class CaseNameExistsInDatabase(Requirement):
    """
    Requirement wrapper for Mellea that ensures case citations in LLM output
    refer to real cases in the provided metadata database.
    """
    # is this taking in the right parameters?
    def __init__(self, case_metadata: list[dict]):
        self._case_metadata = case_metadata
        super().__init__(
            description="The case name should exist in the provided case metadata database.",
            validation_fn=lambda ctx: citation_exists(ctx, self._case_metadata),
        )

# endregion