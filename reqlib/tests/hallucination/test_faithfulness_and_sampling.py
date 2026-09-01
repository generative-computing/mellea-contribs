"""Unit and integration tests for faithfulness_requirement and repair_strategy modules."""

from mellea_contribs.reqlib.stdlib.reqlib.faithfulness_requirement import (
    extract_faithfulness_score,
    extract_unsupported_claims,
    check_faithfulness,
    FaithfulnessRequirement,
)
from mellea_contribs.reqlib.stdlib.reqlib.repair_strategy import (
    SimpleContextGuidedRepairStrategy,
    _DEFAULT_REPAIR_TEMPLATE,
    _REPAIR_TEMPLATE_V2,
)
from mellea.stdlib.requirements import ValidationResult


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _MockOutput:
    """Minimal stand-in for a ``ComputedModelOutputThunk``."""

    def __init__(self, value: str):
        self.value = value

    def __str__(self) -> str:
        return self.value


class _MockContext:
    """Minimal stand-in for a Mellea ``Context``."""

    def __init__(self, text: str):
        self._text = text

    def last_output(self):
        return _MockOutput(self._text)


class _MockRequirement:
    description = "mock requirement"


# ---------------------------------------------------------------------------
# extract_faithfulness_score
# ---------------------------------------------------------------------------


class TestExtractFaithfulnessScore:
    def test_valid_scores(self):
        for score in range(1, 6):
            xml = f"<faithfulness_score>{score}</faithfulness_score>"
            assert extract_faithfulness_score(xml) == str(score)

    def test_score_with_whitespace(self):
        xml = "<faithfulness_score>  4  </faithfulness_score>"
        assert extract_faithfulness_score(xml) == "4"

    def test_missing_tag_returns_none(self):
        assert extract_faithfulness_score("no tag here") is None

    def test_out_of_range_returns_none(self):
        assert extract_faithfulness_score("<faithfulness_score>6</faithfulness_score>") is None
        assert extract_faithfulness_score("<faithfulness_score>0</faithfulness_score>") is None

    def test_score_embedded_in_full_xml(self):
        xml = """
        <FaithfulnessEvaluation>
            <faithfulness_score>2</faithfulness_score>
            <faithfulness_explanation></faithfulness_explanation>
        </FaithfulnessEvaluation>
        """
        assert extract_faithfulness_score(xml) == "2"


# ---------------------------------------------------------------------------
# extract_unsupported_claims
# ---------------------------------------------------------------------------


class TestExtractUnsupportedClaims:
    def test_single_claim(self):
        xml = """
        <UnsupportedClaim>
            <unsupported_claim>The sky is neon green.</unsupported_claim>
            <actual_claim>The sky is blue.</actual_claim>
            <should_be_replaced_by>blue</should_be_replaced_by>
        </UnsupportedClaim>
        """
        claims = extract_unsupported_claims(xml)
        assert len(claims) == 1
        assert claims[0]["unsupported_claim"] == "The sky is neon green."
        assert claims[0]["actual_claim"] == "The sky is blue."
        assert claims[0]["should_be_replaced_by"] == "blue"

    def test_multiple_claims(self):
        xml = """
        <UnsupportedClaim>
            <unsupported_claim>Claim A</unsupported_claim>
            <actual_claim>Fact A</actual_claim>
            <should_be_replaced_by>Fact A</should_be_replaced_by>
        </UnsupportedClaim>
        <UnsupportedClaim>
            <unsupported_claim>Claim B</unsupported_claim>
            <actual_claim>not mentioned</actual_claim>
            <should_be_replaced_by>remove</should_be_replaced_by>
        </UnsupportedClaim>
        """
        claims = extract_unsupported_claims(xml)
        assert len(claims) == 2
        assert claims[1]["should_be_replaced_by"] == "remove"

    def test_no_claims_returns_empty_list(self):
        assert extract_unsupported_claims("<faithfulness_score>5</faithfulness_score>") == []

    def test_missing_inner_tags_default_to_empty_string(self):
        xml = "<UnsupportedClaim><unsupported_claim>only this</unsupported_claim></UnsupportedClaim>"
        claims = extract_unsupported_claims(xml)
        assert len(claims) == 1
        assert claims[0]["unsupported_claim"] == "only this"
        assert claims[0]["actual_claim"] == ""
        assert claims[0]["should_be_replaced_by"] == ""


# ---------------------------------------------------------------------------
# check_faithfulness
# ---------------------------------------------------------------------------


class TestCheckFaithfulness:
    """Unit tests using a mock evaluator session."""

    def _make_mock_session(self, response_text: str):
        """Return an object that mimics a MelleaSession for instruct()."""

        class _FakeResult:
            value = response_text

        class _FakeSession:
            def instruct(self, prompt):
                return _FakeResult()

        return _FakeSession()

    def test_high_score_returns_valid(self):
        xml = "<faithfulness_score>5</faithfulness_score>"
        m = self._make_mock_session(xml)
        ctx = _MockContext("Some summary.")
        result = check_faithfulness(ctx, m, transcript_text="Some transcript.")
        assert bool(result) is True

    def test_low_score_returns_invalid_with_reason(self):
        xml = """
        <faithfulness_score>2</faithfulness_score>
        <UnsupportedClaim>
            <unsupported_claim>Bad claim.</unsupported_claim>
            <actual_claim>not mentioned</actual_claim>
            <should_be_replaced_by>remove</should_be_replaced_by>
        </UnsupportedClaim>
        """
        m = self._make_mock_session(xml)
        ctx = _MockContext("Summary with bad claim.")
        result = check_faithfulness(ctx, m, transcript_text="Transcript text.")
        assert bool(result) is False
        assert "Faithfulness Check Failed" in result.reason
        assert "Bad claim." in result.reason

    def test_score_exactly_at_threshold_fails(self):
        """Score equal to threshold (3) should fail."""
        xml = """
        <faithfulness_score>3</faithfulness_score>
        <UnsupportedClaim>
            <unsupported_claim>Borderline claim.</unsupported_claim>
            <actual_claim>not mentioned</actual_claim>
            <should_be_replaced_by>remove</should_be_replaced_by>
        </UnsupportedClaim>
        """
        m = self._make_mock_session(xml)
        ctx = _MockContext("Summary.")
        result = check_faithfulness(ctx, m, transcript_text="Transcript.", threshold=3)
        assert bool(result) is False

    def test_inconclusive_llm_response_passes(self):
        """If the evaluator response contains no parseable score, treat as inconclusive (pass)."""
        m = self._make_mock_session("no tags here")
        ctx = _MockContext("Any summary.")
        result = check_faithfulness(ctx, m, transcript_text="Transcript.")
        assert bool(result) is True

    def test_llm_exception_passes(self):
        """If the evaluator session raises, treat as inconclusive (pass)."""

        class _ErrorSession:
            def instruct(self, prompt):
                raise RuntimeError("network error")

        ctx = _MockContext("Any summary.")
        result = check_faithfulness(ctx, _ErrorSession(), transcript_text="Transcript.")
        assert bool(result) is True

    def test_custom_threshold(self):
        """Score of 4 should fail when threshold=4."""
        xml = """
        <faithfulness_score>4</faithfulness_score>
        <UnsupportedClaim>
            <unsupported_claim>Minor claim.</unsupported_claim>
            <actual_claim>not mentioned</actual_claim>
            <should_be_replaced_by>remove</should_be_replaced_by>
        </UnsupportedClaim>
        """
        m = self._make_mock_session(xml)
        ctx = _MockContext("Summary.")
        result = check_faithfulness(ctx, m, transcript_text="Transcript.", threshold=4)
        assert bool(result) is False


# ---------------------------------------------------------------------------
# FaithfulnessRequirement
# ---------------------------------------------------------------------------


class TestFaithfulnessRequirement:
    def _make_mock_session(self, response_text: str):
        class _FakeResult:
            value = response_text

        class _FakeSession:
            def instruct(self, prompt):
                return _FakeResult()

        return _FakeSession()

    def test_requirement_passes_when_faithful(self):
        m = self._make_mock_session("<faithfulness_score>5</faithfulness_score>")
        req = FaithfulnessRequirement(evaluator_session=m, transcript_text="transcript")
        ctx = _MockContext("faithful summary")
        # validation_fn is the synchronous entry point; validate() is async and
        # requires a backend — call the fn directly to keep the test hermetic.
        result = req.validation_fn(ctx)
        assert bool(result) is True

    def test_requirement_fails_when_hallucinated(self):
        xml = """
        <faithfulness_score>1</faithfulness_score>
        <UnsupportedClaim>
            <unsupported_claim>Invented fact.</unsupported_claim>
            <actual_claim>not mentioned</actual_claim>
            <should_be_replaced_by>remove</should_be_replaced_by>
        </UnsupportedClaim>
        """
        m = self._make_mock_session(xml)
        req = FaithfulnessRequirement(evaluator_session=m, transcript_text="transcript")
        ctx = _MockContext("summary with invented fact")
        result = req.validation_fn(ctx)
        assert bool(result) is False
        assert "Invented fact." in result.reason

    def test_description_is_set(self):
        m = self._make_mock_session("")
        req = FaithfulnessRequirement(evaluator_session=m, transcript_text="t")
        assert "faithful" in req.description.lower()


# ---------------------------------------------------------------------------
# SimpleContextGuidedRepairStrategy
# ---------------------------------------------------------------------------


class TestSimpleContextGuidedRepairStrategy:
    """Unit tests for the repair strategy using stub components."""

    def _make_strategy(self, template=None):
        return SimpleContextGuidedRepairStrategy(loop_budget=3, repair_template=template)

    # --- select_from_failure ---

    def test_select_from_failure_returns_best_index(self):
        # attempt 0: 1 pass, attempt 1: 2 passes
        val = [
            [(_MockRequirement(), ValidationResult(True)), (_MockRequirement(), ValidationResult(False))],
            [(_MockRequirement(), ValidationResult(True)), (_MockRequirement(), ValidationResult(True))],
        ]
        idx = SimpleContextGuidedRepairStrategy.select_from_failure([], [], val)
        assert idx == 1

    def test_select_from_failure_empty_returns_zero(self):
        assert SimpleContextGuidedRepairStrategy.select_from_failure([], [], []) == 0

    def test_select_from_failure_all_fail(self):
        val = [
            [(_MockRequirement(), ValidationResult(False))],
            [(_MockRequirement(), ValidationResult(False))],
        ]
        # both are equal; max will pick first
        idx = SimpleContextGuidedRepairStrategy.select_from_failure([], [], val)
        assert idx in (0, 1)

    # --- default template used when none provided ---

    def test_default_template_is_used(self):
        strategy = SimpleContextGuidedRepairStrategy(loop_budget=2)
        assert strategy._repair_template is _DEFAULT_REPAIR_TEMPLATE

    def test_custom_template_stored(self):
        strategy = SimpleContextGuidedRepairStrategy(
            loop_budget=2, repair_template=_REPAIR_TEMPLATE_V2
        )
        assert strategy._repair_template is _REPAIR_TEMPLATE_V2

    # --- repair: non-Instruction action is returned unchanged ---

    def test_repair_non_instruction_passthrough(self):
        strategy = self._make_strategy()

        class _NotInstruction:
            pass

        action = _NotInstruction()
        old_ctx = object()
        returned_action, returned_ctx = strategy.repair(
            old_ctx=old_ctx,
            new_ctx=object(),
            past_actions=[action],
            past_results=[_MockOutput("response")],
            past_val=[[]],
        )
        assert returned_action is action
        assert returned_ctx is old_ctx

    # --- repair: Instruction action gets repair string injected ---

    def test_repair_instruction_injects_feedback(self):
        from mellea.stdlib.components import Instruction

        strategy = self._make_strategy()

        req = _MockRequirement()
        val_result = ValidationResult(False, reason="score too low")

        action = Instruction("Summarize the transcript.")
        old_ctx = object()

        returned_action, returned_ctx = strategy.repair(
            old_ctx=old_ctx,
            new_ctx=object(),
            past_actions=[action],
            past_results=[_MockOutput("my response")],
            past_val=[[(req, val_result)]],
        )

        # copy_and_repair returns a new Instruction with _repair_string set
        assert isinstance(returned_action, Instruction)
        assert returned_action._repair_string is not None
        assert "my response" in returned_action._repair_string
        assert "score too low" in returned_action._repair_string
        assert "mock requirement" in returned_action._repair_string
        # Must return old_ctx, not new_ctx
        assert returned_ctx is old_ctx

    def test_repair_uses_old_ctx(self):
        """repair() must return old_ctx so SimpleContext stays stateless."""
        strategy = self._make_strategy()

        class _MockInstruction:
            def copy_and_repair(self, repair_string):
                return self

        old_ctx = object()
        new_ctx = object()

        _, returned_ctx = strategy.repair(
            old_ctx=old_ctx,
            new_ctx=new_ctx,
            past_actions=[_MockInstruction()],
            past_results=[_MockOutput("r")],
            past_val=[[]],
        )
        assert returned_ctx is old_ctx

    def test_repair_v2_template_contains_faithfulness_guidance(self):
        assert "should_be_replaced_by" in _REPAIR_TEMPLATE_V2
        assert "unsupported" in _REPAIR_TEMPLATE_V2.lower()
        assert "remove" in _REPAIR_TEMPLATE_V2
