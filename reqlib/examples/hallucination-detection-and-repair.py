# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mellea"
# ]
# ///
import logging
import os

from mellea import start_session
from mellea.stdlib.requirements import req

from mellea_contribs.reqlib.stdlib.reqlib.faithfulness_requirement import (
    check_faithfulness,
)
from mellea_contribs.reqlib.stdlib.reqlib.repair_strategy import (
    REPAIR_TEMPLATE_V2,
    SimpleContextGuidedRepairStrategy,
)

# -- Constants
LOOP_BUDGET = 2

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_TRANSCRIPT = """
Alice: Let's kick off the Q3 planning meeting. First topic: the new billing API.
Bob: The billing API migration is on track. We're targeting a rollout on September 15th.
Alice: Good. Second topic: customer support headcount.
Carol: We're hiring two additional support engineers, starting next month.
Bob: Also, we decided to postpone the mobile app redesign until Q4 due to budget constraints.
Alice: Agreed. Let's revisit the mobile redesign in the Q4 planning meeting.
"""


def main():
    # -- Two independent sessions: one drives summarisation, the other
    #    is used exclusively by the faithfulness requirement check so
    #    that each concern has its own conversation history / state.
    with (
        start_session(role="summarisation") as summarisation_session,
        start_session(role="requirement") as requirement_session,
    ):
        logger.info("Sessions ready; starting faithfulness-checked summarization")

        faithfulness_requirement = req(
            "The summary must be faithful to the meeting transcript.",
            validation_fn=lambda ctx: check_faithfulness(
                ctx, requirement_session, transcript_text=SAMPLE_TRANSCRIPT
            ),
        )

        result = summarisation_session.instruct(
            "Summarize the following meeting transcript in 3-4 sentences, "
            "capturing only what was actually discussed.",
            grounding_context={"transcript": SAMPLE_TRANSCRIPT},
            requirements=[faithfulness_requirement],
            strategy=SimpleContextGuidedRepairStrategy(
                loop_budget=LOOP_BUDGET, repair_template=REPAIR_TEMPLATE_V2
            ),
            return_sampling_results=True,
        )

        attempts = len(result.sample_generations)
        if result.success:
            logger.info("Faithfulness check passed after %d attempt(s)", attempts)
        else:
            logger.warning(
                "Faithfulness check did not pass within loop_budget=%d "
                "(%d attempt(s) made); returning best available summary",
                LOOP_BUDGET,
                attempts,
            )

        summary = str(result.result)
        logger.debug("Final summary: %s", summary)
        print(summary)


if __name__ == "__main__":
    main()
