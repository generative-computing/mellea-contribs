from mellea_contribs.reqlib.check_AL_statutes import parse_AL, check_AL, validate_AL_statutes, get_AL_statutes

from mellea import start_session
from mellea.stdlib.requirement import req
from mellea.stdlib.sampling import RejectionSamplingStrategy

def test_parse_AL():
    text = """Ala. Code §1-9-20 (2020) Ala. Code §10A-21-8 (1999)
    Ala. Code §2-7A-3 (2024)"""

    assert(check_AL(parse_AL(text)) == [False, False, True])

    text = """ """
    assert(check_AL(parse_AL(text)) == [])

    text = """Ala. Code §8-5-3 (2021) Ala. Code §6-6-26.01 (2018)"""
    assert(check_AL(parse_AL(text)) == [False, True])


def test_validate_AL_statutes():
    m = start_session()
    generate_AL_statutes = m.instruct(
        "Generate a list of Alabama statute citations with a title number between 1 and 10A, inclusive, according to Bluebook format, namely 'Ala. Code §[title]-[section]-[rest] (year)'. Ensure that at all citations are valid and exist in the Alabama Code.",
        requirements=[req("The result should be a list of valid Alabama statute citations", validation_fn=lambda ctx: validate_AL_statutes(get_AL_statutes(ctx)))],
        strategy=RejectionSamplingStrategy(loop_budget=5),
        return_sampling_results=True,
    )
    print("Generated Alabama Statutes:", generate_AL_statutes.value)
    assert generate_AL_statutes.success

test_validate_AL_statutes()
