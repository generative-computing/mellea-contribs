import pytest

import os

from mellea_contribs.reqlib.is_appellate_case import load_jsons_from_folder, get_court_from_case, is_appellate_court

from mellea import start_session
from mellea.stdlib.requirement import req
from mellea.stdlib.sampling import RejectionSamplingStrategy


def test_is_appellate_court():
    assert is_appellate_court("Supreme Court of New Jersey").as_bool()
    assert not is_appellate_court("Tax Court of New Jersey").as_bool()
    assert is_appellate_court("Pennsylvania Commonwealth Court").as_bool()
    assert is_appellate_court("U.S. Court of Appeals for the First Circuit").as_bool()
    assert is_appellate_court("Maryland Appellate Court").as_bool()
    assert not is_appellate_court("District Court of Maryland").as_bool()


def test_appellate_case_session():
    case_name = "ARTHUR DeMOORS, PLAINTIFF-RESPONDENT, v. ATLANTIC CASUALTY INSURANCE COMPANY OF NEWARK, NEW JERSEY, A CORPORATION, DEFENDANT-APPELLANT"
    folder_path = os.path.join(os.path.dirname(__file__), "data", "legal", "nj_case_metadata")
    folder_path = os.path.normpath(folder_path)

    m = start_session()
    case_metadata = load_jsons_from_folder(folder_path)
    appellate_case = m.instruct(
        f"Return the following string (only return the characters after the colon, no other words): {case_name}",
        requirements=[req("The result should be an appellate court case", validation_fn=lambda ctx: is_appellate_court(get_court_from_case(ctx, case_metadata)))],
        strategy=RejectionSamplingStrategy(loop_budget=5),
        return_sampling_results=True,
    )
    assert appellate_case.success