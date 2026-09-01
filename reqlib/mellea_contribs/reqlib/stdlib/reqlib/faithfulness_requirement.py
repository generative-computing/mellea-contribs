"""Faithfulness requirement and helpers for Mellea agents.

Purpose: Detects hallucinations in LLM-generated summaries by comparing them
against a source transcript using an LLM-as-a-judge pattern.

What it does:
- Prompts a secondary ``MelleaSession`` to score the faithfulness of an output
  against a grounding transcript (1–5 scale).
- Extracts unsupported claims (with suggested replacements) from the evaluator
  response.
- Returns a :class:`~mellea.stdlib.requirements.ValidationResult` that
  ``Requirement``-based repair strategies (e.g.
  :class:`~mellea_contribs.reqlib.stdlib.reqlib.repair_strategy.SimpleContextGuidedRepairStrategy`)
  can act on.

When should an agent use it?
- When the agent produces textual summaries that must remain faithful to a
  provided source document or transcript.
- When hallucination detection and guided repair are needed in the same
  instruct-validate-repair loop.
"""

from __future__ import annotations

import logging
import re

from mellea import MelleaSession
from mellea.stdlib.requirements import Requirement, ValidationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------


def extract_faithfulness_score(xml_string: str) -> str | None:
    """Extract the integer score from a ``<faithfulness_score>`` tag.

    Args:
        xml_string: Raw XML text returned by the faithfulness evaluator LLM.

    Returns:
        The score as a single-character string (``"1"``–``"5"``), or ``None``
        if the tag is absent or the value is out of range.
    """
    pattern = r"<faithfulness_score>\s*([1-5])\s*</faithfulness_score>"
    match = re.search(pattern, xml_string)
    return match.group(1) if match else None


def extract_unsupported_claims(xml_string: str) -> list[dict[str, str]]:
    """Extract all ``<UnsupportedClaim>`` blocks from the evaluator response.

    Args:
        xml_string: Raw XML text returned by the faithfulness evaluator LLM.

    Returns:
        A list of dicts, each with keys ``unsupported_claim``, ``actual_claim``,
        and ``should_be_replaced_by``.
    """
    block_pattern = r"<UnsupportedClaim>(.*?)</UnsupportedClaim>"
    blocks = re.findall(block_pattern, xml_string, re.DOTALL)

    claims = []
    for block in blocks:
        unsupported = re.search(
            r"<unsupported_claim>(.*?)</unsupported_claim>", block, re.DOTALL
        )
        actual = re.search(r"<actual_claim>(.*?)</actual_claim>", block, re.DOTALL)
        replaced = re.search(
            r"<should_be_replaced_by>(.*?)</should_be_replaced_by>", block, re.DOTALL
        )
        claims.append(
            {
                "unsupported_claim": unsupported.group(1).strip()
                if unsupported
                else "",
                "actual_claim": actual.group(1).strip() if actual else "",
                "should_be_replaced_by": replaced.group(1).strip() if replaced else "",
            }
        )
    return claims


# ---------------------------------------------------------------------------
# LLM evaluation helpers
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are an expert evaluator assessing faithfulness of a meeting summary against its transcript.

**MEETING TRANSCRIPT:**
{transcript}

**SUMMARY TO EVALUATE:**
{summary}

Assign a faithfulness score (1-5) based on how well every claim in the summary is grounded in the transcript:

- **5 (Perfectly Faithful):** Every claim is directly supported by the transcript.
- **4 (Mostly Faithful):** Core information is correct; minor rephrasing or common-sense fillers present.
- **3 (Partially Faithful):** Some factual content from the transcript, but includes minor inaccuracies or one unverified claim.
- **2 (Low Faithfulness):** Contradicts the transcript or relies heavily on outside knowledge.
- **1 (Not Faithful):** Significant falsehoods, hallucinations, or completely ignores the source material.

For each claim in the summary that is not directly supported by the transcript, produce one entry in
faithfulness_explanation with:
- unsupported_claim: the exact claim from the summary that is not grounded
- actual_claim: what the transcript actually says (or "not mentioned" if absent entirely)
- should_be_replaced_by: corrected text to replace the hallucination, or "remove" if it should be deleted

Output only xml like tags
Output Format Example:
    <FaithfulnessEvaluation>
        <faithfulness_score>3</faithfulness_score>
        <faithfulness_explanation>
            <UnsupportedClaim>
                <unsupported_claim>The sky is neon green.</unsupported_claim>
                <actual_claim>The sky is blue.</actual_claim>
                <should_be_replaced_by>blue</should_be_replaced_by>
            </UnsupportedClaim>
        </faithfulness_explanation>
    </FaithfulnessEvaluation>
"""


def capture_hallucinated_items(
    m: MelleaSession, transcript: str, summary: str
) -> str | None:
    """Run the faithfulness-evaluation prompt and return the raw XML response.

    Args:
        m: A ``MelleaSession`` dedicated to the evaluator role (should be
           kept separate from the summarisation session so that each has its
           own conversation history).
        transcript: The source document the summary should be faithful to.
        summary: The LLM-generated text to evaluate.

    Returns:
        The raw XML string from the evaluator, or ``None`` if the LLM call
        fails (callers should treat ``None`` as an inconclusive evaluation).
    """
    prompt = _FAITHFULNESS_PROMPT.format(transcript=transcript, summary=summary)
    try:
        out = m.instruct(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Faithfulness evaluation LLM call failed: %s", exc)
        return None
    return out.value


def _evaluate_faithfulness(
    m: MelleaSession,
    transcript_text: str,
    summary: str,
    threshold: int = 3,
) -> list[dict[str, str]] | None:
    """Evaluate summary faithfulness against the transcript.

    Args:
        m: Evaluator ``MelleaSession``.
        transcript_text: Source transcript.
        summary: Generated summary to check.
        threshold: Scores *above* this value are considered faithful (default
            ``3``). Scores at or below trigger a failure.

    Returns:
        ``None`` if the summary is faithful (score above threshold) or the
        evaluation was inconclusive. Otherwise returns the list of unsupported
        claims extracted from the evaluator response.
    """
    hallucinated = capture_hallucinated_items(
        m, transcript=transcript_text, summary=summary
    )
    if hallucinated is None:
        return None

    score = extract_faithfulness_score(hallucinated)
    if score is None:
        return None

    if int(score) > threshold:
        return None

    return extract_unsupported_claims(hallucinated)


# ---------------------------------------------------------------------------
# Public requirement function and Requirement class
# ---------------------------------------------------------------------------


def check_faithfulness(
    ctx, m: MelleaSession, transcript_text: str, threshold: int = 3
) -> ValidationResult:
    """Requirement validation function: check that the last output is faithful.

    Intended to be used as the ``validation_fn`` of a ``req(...)`` call or
    :class:`FaithfulnessRequirement`.

    Args:
        ctx: Mellea runtime context exposing ``last_output()``.
        m: A ``MelleaSession`` used exclusively for faithfulness evaluation.
        transcript_text: The source transcript the summary must be grounded in.
        threshold: Faithfulness score threshold (inclusive); outputs scoring at
            or below this value fail. Defaults to ``3``.

    Returns:
        :class:`~mellea.stdlib.requirements.ValidationResult` — ``True`` when
        the output is faithful, ``False`` with a reason string listing the
        unsupported claims when it is not.
    """
    summary = str(ctx.last_output())

    explanations = _evaluate_faithfulness(
        m=m,
        transcript_text=transcript_text,
        summary=summary,
        threshold=threshold,
    )
    if explanations is None:
        return ValidationResult(True)

    reason = "Faithfulness Check Failed " + str(explanations)
    return ValidationResult(False, reason=reason)


class FaithfulnessRequirement(Requirement):
    """Requirement that ensures the LLM output is faithful to a source transcript.

    Wraps :func:`check_faithfulness` as a reusable :class:`Requirement` object.

    Example::

        from mellea_contribs.reqlib.stdlib.reqlib.faithfulness_requirement import (
            FaithfulnessRequirement,
        )

        faithfulness_req = FaithfulnessRequirement(
            evaluator_session=requirement_session,
            transcript_text=SAMPLE_TRANSCRIPT,
        )
        result = session.instruct(
            "Summarize ...",
            requirements=[faithfulness_req],
            strategy=SimpleContextGuidedRepairStrategy(loop_budget=3),
        )
    """

    def __init__(
        self,
        evaluator_session: MelleaSession,
        transcript_text: str,
        threshold: int = 3,
    ):
        """
        Args:
            evaluator_session: A ``MelleaSession`` used exclusively for
                faithfulness evaluation (keep separate from the generation
                session).
            transcript_text: The source document the LLM output must be
                grounded in.
            threshold: Faithfulness score threshold (inclusive); outputs
                scoring at or below this value fail. Defaults to ``3``.
        """
        self._evaluator_session = evaluator_session
        self._transcript_text = transcript_text
        self._threshold = threshold
        super().__init__(
            description="The output must be faithful to the provided transcript.",
            validation_fn=lambda ctx: check_faithfulness(
                ctx,
                m=self._evaluator_session,
                transcript_text=self._transcript_text,
                threshold=self._threshold,
            ),
        )
