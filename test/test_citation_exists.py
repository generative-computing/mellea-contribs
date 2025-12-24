import json
import pytest
from unittest.mock import patch, MagicMock

from mellea.stdlib.base import Context
from mellea_contribs.reqlib.citation_exists import (
    text_to_urls,
    citation_exists,
    extract_case_metadata_url,
)

class MockContext:
    """
    Minimal mock of Mellea Context API.
    """
    def __init__(self, value):
        self._value = value

    def last_output(self):
        # last_output() should return a string, not an object
        return self._value
    
@pytest.fixture
def database():
    """
    Load the real metadata DB.
    """
    db_path = "/Users/anooshkapendyal/mellea-contribs/test/citation_exists_database.json"
    with open(db_path) as f:
        return json.load(f)


# region text_to_urls tests

def test_text_to_urls_basic_extraction():
    """
    Verifies that text_to_urls correctly extracts URL values from citation objects
    when the Citator returns a well-formed citation containing a valid .URL attribute.
    """
    mock_citation = MagicMock()
    mock_citation.URL = "https://cite.case.law/us/123/456"

    with patch("mellea_contribs.reqlib.citation_exists.Citator") as cit:
        cit.return_value.list_cites.return_value = [mock_citation]
        urls = text_to_urls("Example text")

    assert urls == ["https://cite.case.law/us/123/456"]


def test_text_to_urls_missing_url_attribute():
    """
    Ensures that text_to_urls returns a failing ValidationResult when a citation
    is missing its required .URL attribute and that the resulting error message is informative.
    """
    bad_cite = MagicMock()
    del bad_cite.URL  # simulate missing URL

    with patch("mellea_contribs.reqlib.citation_exists.Citator") as cit:
        cit.return_value.list_cites.return_value = [bad_cite]
        result = text_to_urls("text")

    assert result.as_bool() is False
    assert "no url attribute" in result.reason.lower()


def test_text_to_urls_empty_text():
    """
    Checks that text_to_urls correctly returns an empty list when no citations are detected in the input text.
    """
    with patch("mellea_contribs.reqlib.citation_exists.Citator") as cit:
        cit.return_value.list_cites.return_value = []
        urls = text_to_urls("")

    assert urls == []

# endregion


# region extract_case_metadata_url test

def test_extract_case_metadata_url_nonstandard():
    """
    Verifies that extract_case_metadata_url works even if the original URL goes to a specific
    section of the page.
    """
    test_url = "https://case.law/caselaw/?reporter=us&volume=477&case=0561-01#p574"
    result = extract_case_metadata_url(test_url)
    assert result == "https://static.case.law/us/477/cases/0561-01.json"


def test_extract_case_metadata_url_bad():
    """
    Ensures that invalid URLs fail gracefully.
    """
    test_url = "https://shmeegus.com/"

    # Mock the playwright context manager
    with patch("mellea_contribs.reqlib.citation_exists.sync_playwright") as mock_pw:
        mock_browser = MagicMock()
        mock_page = MagicMock()

        # Configure playwright mock chain
        mock_pw.return_value.__enter__.return_value.chromium.launch.return_value = mock_browser
        mock_browser.new_page.return_value = mock_page

        # Simulate "Download case metadata" link NOT found
        mock_page.wait_for_selector.return_value = None

        result = extract_case_metadata_url(test_url)

    assert result.as_bool() is False
    assert "no metadata link found on page" in result.reason.lower()

# endregion


# region citation_exists tests (context failures)

def test_citation_exists_no_context(database):
    """
    Verifies that the citation_exists function correctly handles the case when no context is provided.
    """
    result = citation_exists(None, database)
    assert result.as_bool() is False
    assert "no context" in result.reason.lower()


def test_citation_exists_last_output_none(database):
    """
    Checks that citation_exists correctly handles a Context whose last_output() method returns None.
    """
    ctx = MockContext(None)
    result = citation_exists(ctx, database)
    assert result.as_bool() is False
    assert "no last output" in result.reason.lower()


def test_citation_exists_last_output_not_string(database):
    """
    Ensures that citation_exists corrrectly handles a Context whose last_output() returns a non-string value.
    """
    ctx = MockContext(123)  # not a string
    with pytest.raises(TypeError):
        citation_exists(ctx, database)

# endregion


# region citation_exists tests (no citations)

def test_citation_exists_no_citations(database):
    """
    Verifies that citation_exists correctly handles text that contains no citations.
    """
    ctx = MockContext("text with no citations")

    with patch("mellea_contribs.reqlib.citation_exists.text_to_urls") as text_to_url:
        text_to_url.return_value = []
        result = citation_exists(ctx, database)

    assert result.as_bool() is True
    assert "no citations found" in result.reason.lower()

# endregion


# region citation_exists tests (main validation logic)

def test_citation_exists_case_found(database):
    """
    Ensures that the function correctly returns a passing ValidationResult when
    a cited case is valid and present in the database.
    """
    ctx = MockContext("Some citation")

    with patch("mellea_contribs.reqlib.citation_exists.text_to_urls") as text_to_url, \
         patch("mellea_contribs.reqlib.citation_exists.extract_case_metadata_url") as extracted_metadata_url, \
         patch("mellea_contribs.reqlib.citation_exists.metadata_url_to_json") as meta:

        text_to_url.return_value = ["https://cite.case.law/us/111/222"]
        extracted_metadata_url.return_value = "https://cite.case.law/us/111/222/metadata.json"

        # Pick a real ID from the database
        real_id = next(iter({d["id"] for d in database}))
        meta.return_value = {"id": real_id}

        result = citation_exists(ctx, database)

    assert result.as_bool() is True


def test_citation_exists_case_missing_from_db(database):
    """
    Ensures that the function correctly returns a failing ValidationResult when
    a cited case is not present in the database.
    """
    ctx = MockContext("Some citation")

    with patch("mellea_contribs.reqlib.citation_exists.text_to_urls") as text_to_url, \
         patch("mellea_contribs.reqlib.citation_exists.extract_case_metadata_url") as extracted_metadata_url, \
         patch("mellea_contribs.reqlib.citation_exists.metadata_url_to_json") as meta:

        text_to_url.return_value = ["https://cite.case.law/us/333/444"]
        extracted_metadata_url.return_value = "https://cite.case.law/us/333/444/metadata.json"
        meta.return_value = {"id": "NON_EXISTENT_CASE_ID"}

        result = citation_exists(ctx, database)

    assert result.as_bool() is False
    assert "not found" in result.reason.lower()


def test_citation_exists_non_caselaw_url_ignored(database):
    """
    Checks that non cite.case.law URLs are ignored.
    """
    ctx = MockContext("Some citation")

    with patch("mellea_contribs.reqlib.citation_exists.text_to_urls") as text_to_url:
        text_to_url.return_value = ["https://www.law.cornell.edu/uscode/text/42/1988"]

        result = citation_exists(ctx, database)

    # Should pass because the only citation is non-caselaw
    assert result.as_bool() is True


def test_citation_exists_real_case_law_case(database):
    """
    Tests using a real case.law URL.
    """
    ctx = MockContext(
        "See Smith v. State, 154 Ala. 1 (1907)."
    )

    real_case_url = (
        "https://case.law/caselaw/?reporter=ala&volume=154&case=0001-01"
    )

    real_metadata = {
        "id": 5668189,
        "name": "Smith v. State",
        "citations": ["154 Ala. 1"],
    }

    with patch("mellea_contribs.reqlib.citation_exists.text_to_urls") as text_to_url, \
         patch("mellea_contribs.reqlib.citation_exists.requests.get") as mock_get:

        text_to_url.return_value = [real_case_url]

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = real_metadata
        mock_get.return_value = mock_resp

        result = citation_exists(ctx, database)

    assert result.as_bool() is True

# endregion