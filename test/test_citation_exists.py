import json
import pytest
from unittest.mock import patch, MagicMock

from mellea.stdlib.base import Context
from mellea_contribs.reqlib.citation_exists import *

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
    
@pytest.fixture
def small_database():
    """
    Load the real metadata DB (small version).
    """
    small_db_path = "/Users/anooshkapendyal/mellea-contribs/test/small_citation_exists_database.json"
    with open(small_db_path) as f:
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

    assert list(urls.keys()) == ["https://cite.case.law/us/123/456"]


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
    Checks that text_to_urls correctly returns an empty dictionary when no citations are
    detected in the input text.
    """
    with patch("mellea_contribs.reqlib.citation_exists.Citator") as cit:
        cit.return_value.list_cites.return_value = []
        urls = text_to_urls("")

    assert urls == {}

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

# region collect_ids test

def test_collect_ids_in_database(small_database):
    """
    Tests helper to collect all case IDs in database.
    """
    result = collect_ids_in_database(small_database)
    assert result == {6714178, 6714194}
    
# endregion


# region parse_db_cite tests

@pytest.mark.parametrize(
    "cite,expected",
    [
        ("1 Idaho 425", ("1", "idaho", "425")),
        ("1 Va. Cas. 109", ("1", "va cas", "109")),
        ("2 Am. Dec. 564", ("2", "am dec", "564")),
        ("1 Wash. 128", ("1", "wash", "128")),
        ("1 Paige Ch. 384", ("1", "paige ch", "384")),
        ("3 Call 424", ("3", "call", "424")),
        ("7 Va. 418", ("7", "va", "418")),
    ],
)

def test_parse_db_cite_valid_cases(cite, expected):
    """
    Verifies that parse_db_cite correctly parses volume, normalized reporter,
    and page for well-formed case citations.
    """
    assert parse_db_cite(cite) == expected


def test_parse_db_cite_too_short():
    """
    Checks that citations with fewer than 3 components should return None.
    """
    assert parse_db_cite("1 Va.") is None
    assert parse_db_cite("123") is None
    assert parse_db_cite("") is None


def test_parse_db_cite_extra_whitespace():
    """
    Ensures that extra whitespace should not affect parsing.
    """
    cite = "  1   Va.   Cas.   109  "
    assert parse_db_cite(cite) == ("1", "va cas", "109")


# endregion


# region build_citation_index tests

def test_build_citation_index_small_database(small_database):
    """
    Verifies that build_citation_index correctly extracts and normalizes
    all citations from the small test database.
    """
    result = build_citation_index(small_database)

    expected = {
        ("1", "wash", "1"),
        ("1", "va", "1"),
        ("1", "wash", "4"),
        ("1", "va", "4"),
    }

    assert result == expected


def test_build_citation_index_ignores_unparseable_citations():
    """
    Ensures malformed citations do not break index construction.
    """
    bad_db = [
        {"citations": [{"cite": "Invalid"}]},
        {"citations": [{"cite": "1 Va. 1"}]},
    ]

    result = build_citation_index(bad_db)

    assert result == {("1", "va", "1")}

# endregion


# region non_caselaw_citation_exists tests

class MockEyeciteCitation:
    def __init__(self, volume, reporter, page):
        self.groups = {
            "volume": volume,
            "reporter": reporter,
            "page": page,
        }


def test_non_caselaw_citation_exists_match_found(small_database):
    """
    Single eyecite citation that matches a DB citation.
    """
    mock_cite = MockEyeciteCitation("1", "Wash.", "1")

    with patch("mellea_contribs.reqlib.citation_exists.get_citations") as mock_get:
        mock_get.return_value = [mock_cite]
        result = non_caselaw_citation_exists("1 Wash. 1", small_database)

    assert result is True


def test_non_caselaw_citation_exists_no_match(small_database):
    """
    Single eyecite citation that does NOT match DB.
    """
    mock_cite = MockEyeciteCitation("99", "Wash.", "999")

    with patch("mellea_contribs.reqlib.citation_exists.get_citations") as mock_get:
        mock_get.return_value = [mock_cite]
        result = non_caselaw_citation_exists("99 Wash. 999", small_database)

    assert result is False


def test_non_caselaw_citation_exists_no_citations(small_database):
    """
    Eyecite returns no citations (i.e. hard failure).
    """
    with patch("mellea_contribs.reqlib.citation_exists.get_citations") as mock_get:
        mock_get.return_value = []
        result = non_caselaw_citation_exists("quoting Theatre Enterprises, Inc. v. Paramount Film Distributing Corp., 346 U. S. 537, 541 (1954)", small_database)

    assert isinstance(result, ValidationResult)
    assert result.as_bool() is False
    assert "parsing citations" in result.reason.lower()

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
    Ensures that citation_exists corrrectly handles a Context whose last_output()
    returns a non-string value.
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
        text_to_url.return_value = {}
        result = citation_exists(ctx, database)

    assert result.as_bool() is True
    assert "no citations found" in result.reason.lower()

# endregion


# region citation_exists tests (main validation logic)

# subregion citations with case.law URL

def test_citation_exists_case_found(database):
    """
    Ensures that the function correctly returns a passing ValidationResult when
    a cited case is valid and present in the database.
    """
    ctx = MockContext("Some citation")

    with patch("mellea_contribs.reqlib.citation_exists.text_to_urls") as text_to_url, \
         patch("mellea_contribs.reqlib.citation_exists.extract_case_metadata_url") as extracted_metadata_url, \
         patch("mellea_contribs.reqlib.citation_exists.metadata_url_to_json") as meta:

        text_to_url.return_value = {"https://cite.case.law/us/111/222": "Some citation text"}
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

        text_to_url.return_value = {"https://cite.case.law/us/333/444": "Some citation text"}
        extracted_metadata_url.return_value = "https://cite.case.law/us/333/444/metadata.json"
        meta.return_value = {"id": "NON_EXISTENT_CASE_ID"}

        result = citation_exists(ctx, database)

    assert result.as_bool() is False
    assert "not found" in result.reason.lower()


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

        text_to_url.return_value = {real_case_url: "Smith v. State, 154 Ala. 1 (1907)"}

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = real_metadata
        mock_get.return_value = mock_resp

        result = citation_exists(ctx, database)

    assert result.as_bool() is True

# endsubregion

# subregion citations without case.law URL

def test_citation_exists_non_caselaw_hard_failure(database):
    """
    Verifies that when non_caselaw_citation_exists returns ValidationResult(False),
    it should propagate failure.
    Tests Case 1: Hard failure
    """
    ctx = MockContext("Some citation")

    failure = ValidationResult(False, reason="Eyecite parsing failed")

    with patch("mellea_contribs.reqlib.citation_exists.text_to_urls") as text_to_url, \
         patch("mellea_contribs.reqlib.citation_exists.non_caselaw_citation_exists") as non_case:

        text_to_url.return_value = {
            "https://shmeegus.com/non-caselaw": "bad citation"
        }
        non_case.return_value = failure

        result = citation_exists(ctx, database)

    assert result.as_bool() is False
    assert "eyecite parsing failed" in result.reason.lower()


def test_citation_exists_non_caselaw_deterministic_match(database):
    """
    Tests that if non_caselaw_citation_exists returns True, citation_exists should pass.
    Tests Case 2: deterministic match found 
    """
    ctx = MockContext("Some citation")

    with patch("mellea_contribs.reqlib.citation_exists.text_to_urls") as text_to_url, \
         patch("mellea_contribs.reqlib.citation_exists.non_caselaw_citation_exists") as non_case:

        text_to_url.return_value = {
            "https://shmeegus.com/non-caselaw": "1 Wash. 1"
        }
        non_case.return_value = True

        result = citation_exists(ctx, database)

    assert result.as_bool() is True


def test_citation_exists_non_caselaw_inconclusive_allowed(database):
    """
    non_caselaw_citation_exists returns False (inconclusive), so it should NOT fail.
    Tests Case 3: Inconclusive but do not fail
    """
    ctx = MockContext("Some citation")

    with patch("mellea_contribs.reqlib.citation_exists.text_to_urls") as text_to_url, \
         patch("mellea_contribs.reqlib.citation_exists.non_caselaw_citation_exists") as non_case:

        text_to_url.return_value = {
            "https://shmeegus.com/non-caselaw": "67 Shmeegus. 999"
        }
        non_case.return_value = False

        result = citation_exists(ctx, database)

    assert result.as_bool() is True
    assert "non-case.law citations did not fail" in result.reason.lower()

# endsubregion

# subregion mixed case.law and non-case.law test

def test_citation_exists_mixed_case_and_non_case(database):
    """
    Check that one valid case.law citation and one inconclusive non-case.law citation result
    in a ValidationResult of true.
    """
    ctx = MockContext("Some mixed citations")

    real_id = next(iter({d["id"] for d in database}))

    with patch("mellea_contribs.reqlib.citation_exists.text_to_urls") as text_to_url, \
         patch("mellea_contribs.reqlib.citation_exists.extract_case_metadata_url") as extract, \
         patch("mellea_contribs.reqlib.citation_exists.metadata_url_to_json") as meta, \
         patch("mellea_contribs.reqlib.citation_exists.non_caselaw_citation_exists") as non_case:

        text_to_url.return_value = {
            "https://cite.case.law/us/111/222": "Real case",
            "https://shmeegus.com/non-caselaw": "Ambiguous citation",
        }

        extract.return_value = "https://static.case.law/us/111/cases/222.json"
        meta.return_value = {"id": real_id}
        non_case.return_value = False  # inconclusive

        result = citation_exists(ctx, database)

    assert result.as_bool() is True

# endsubregion

# endregion