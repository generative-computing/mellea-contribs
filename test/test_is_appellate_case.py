import pytest
import os

from mellea_contribs.reqlib.is_appellate_case import (
    load_jsons_from_folder, 
    get_court_from_case, 
    is_appellate_court_fullname,
    court_abbv_from_citation,
    is_appellate_court_abbv,
    is_appellate_case
)

from mellea import start_session
from mellea.stdlib.requirement import req
from mellea.stdlib.sampling import RejectionSamplingStrategy


# ---------- #
# UNIT TESTS #
# ---------- #

def test_is_appellate_court_fullname():
    assert is_appellate_court_fullname("Supreme Court of New Jersey").as_bool()
    assert not is_appellate_court_fullname("Tax Court of New Jersey").as_bool()
    assert is_appellate_court_fullname("Pennsylvania Commonwealth Court").as_bool()
    assert is_appellate_court_fullname("U.S. Court of Appeals for the First Circuit").as_bool()
    assert is_appellate_court_fullname("Maryland Appellate Court").as_bool()
    assert not is_appellate_court_fullname("District Court of Maryland").as_bool()


def test_court_abbv_from_citation():
    # case 1 in function
    assert court_abbv_from_citation("Smith v. Jones, 123 F.3d 456 (9th Cir. 2000)") == "9th Cir."
    assert court_abbv_from_citation("United States v. Smith, 789 F.Supp.2d 123 (S.D.N.Y. 2011)") == "S.D.N.Y."
    assert court_abbv_from_citation("State v. Williams, 234 P.3d 567 (Wash. Ct. App. 2015)") == "Wash. Ct. App."
    assert court_abbv_from_citation("Johnson v. State, 456 So.2d 789 (Fla. Dist. Ct. App. 2010)") == "Fla. Dist. Ct. App."

    # case 2 in function
    assert court_abbv_from_citation("Roe v. Wade, 410 U.S. 113 (1973)") == "U.S."
    assert court_abbv_from_citation("Brown v. Board of Education, 347 U.S. 483 (1954)") == "U.S."
    assert court_abbv_from_citation("People v. Johnson, 45 Cal.App.4th 789 (2020)") == "Cal.App.4th"

    # invalid citation
    with pytest.raises(ValueError):
        court_abbv_from_citation("Not a citation")
    with pytest.raises(ValueError):
        court_abbv_from_citation("9th Cir.")
    with pytest.raises(ValueError):
        court_abbv_from_citation("Supreme Court of California")


def test_is_appellate_court_abbv():
    assert is_appellate_court_abbv("U.S.").as_bool()
    assert is_appellate_court_abbv("S.Ct.").as_bool()
    assert is_appellate_court_abbv("Ala.").as_bool()
    assert is_appellate_court_abbv("Cal.").as_bool()
    assert is_appellate_court_abbv("N.Y.").as_bool()
    assert is_appellate_court_abbv("9th Cir.").as_bool()
    assert is_appellate_court_abbv("D.C. Cir.").as_bool()
    assert is_appellate_court_abbv("1st Cir.").as_bool()
    assert is_appellate_court_abbv("11th Cir.").as_bool()
    assert is_appellate_court_abbv("F.3d").as_bool()
    assert is_appellate_court_abbv("F.2d").as_bool()
    assert not is_appellate_court_abbv("F.Supp.2d").as_bool()
    assert not is_appellate_court_abbv("F.Supp.").as_bool()
    assert not is_appellate_court_abbv("S.D.N.Y.").as_bool()
    assert not is_appellate_court_abbv("N.D. Cal.").as_bool()
    assert not is_appellate_court_abbv("E.D. Va.").as_bool()
    assert not is_appellate_court_abbv("W.D. Tex.").as_bool()
    assert is_appellate_court_abbv("Cal.App.4th").as_bool()
    assert is_appellate_court_abbv("Wash. Ct. App.").as_bool()
    assert is_appellate_court_abbv("Fla. Dist. Ct. App.").as_bool()
    assert not is_appellate_court_abbv("Fed. Cl.").as_bool()
    assert not is_appellate_court_abbv("Ct. Intl. Trade").as_bool()
    assert not is_appellate_court_abbv("Tax Ct.").as_bool()
    assert not is_appellate_court_abbv("Bankr. S.D.N.Y.").as_bool()
    assert is_appellate_court_abbv("N.D.").as_bool()
    assert not is_appellate_court_abbv("N.D. Cal.").as_bool()


# ----------------- #
# INTEGRATION TESTS #
# ----------------- #

folder_path = os.path.join(os.path.dirname(__file__), "data", "legal", "nj_case_metadata")
folder_path = os.path.normpath(folder_path)

class MockContext:
    def __init__(self, input):
        self._input = input

    def last_output(self):
        return type("MockOutput", (), {"value": self._input})()


def test_is_appellate_case():
    test_cases = [
        ("Smith v. Jones, 123 F.3d 456 (9th Cir. 2000)", True),
        ("Roe v. Wade, 410 U.S. 113 (1973)", True),
        ("United States v. Smith, 789 F.Supp.2d 123 (S.D.N.Y. 2011)", False),
        ("People v. Johnson, 45 Cal.App.4th 789 (2020)", True),
        ("ARTHUR DeMOORS, PLAINTIFF-RESPONDENT, v. ATLANTIC CASUALTY INSURANCE COMPANY OF NEWARK, NEW JERSEY, A CORPORATION, DEFENDANT-APPELLANT", True)
    ]
    for input, expected_appellate in test_cases:
        ctx = MockContext(input)
        result = is_appellate_case(ctx, folder_path)
        assert result.as_bool() == expected_appellate, f"Failed for input: {input}"


def test_appellate_case_session():
    case_name = "ARTHUR DeMOORS, PLAINTIFF-RESPONDENT, v. ATLANTIC CASUALTY INSURANCE COMPANY OF NEWARK, NEW JERSEY, A CORPORATION, DEFENDANT-APPELLANT"
    m = start_session()
    appellate_case = m.instruct(
        f"Return the following string (only return the characters after the colon, no other words): {case_name}",
        requirements=[req("The result should be an appellate court case name or citation", validation_fn=lambda ctx: is_appellate_case(ctx, folder_path))],
        strategy=RejectionSamplingStrategy(loop_budget=5),
        return_sampling_results=True,
    )
    assert appellate_case.success