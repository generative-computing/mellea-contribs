import collections
import random
import functools
import itertools
import asyncio
from mellea import MelleaSession
from mellea.backends import Backend
from mellea.stdlib.base import Context
from mellea.stdlib.functional import ainstruct
from mellea.helpers.fancy_logger import FancyLogger
from mellea.helpers.event_loop_helper import _run_async_in_thread

from pydantic import BaseModel

from typing import Literal

from .util import session_wrapper, sync_wrapper

logger = FancyLogger.get_logger()

class YesNo(BaseModel):
    answer : Literal["yes","no"]

async def abool(backend:Backend, ctx:Context, prompt:str, vote:int=3, **kwargs) -> bool:
    """
    Answers a yes/no question.
    """

    if vote % 2 == 0:
        logger.warning(
            "the specified number of votes in a majority vote is even, making ties possible. Increasing the value by one to avoid this."
        )
        vote += 1

    async def fn():

        output, _ = await ainstruct(f"{prompt} Answer yes or no.",
                                    ctx, backend,
                                    format=YesNo, **kwargs)

        yesno = YesNo.model_validate_json(output.value)

        return yesno.answer == "yes"

    tasks = [fn() for _ in range(vote)]
    results = await asyncio.gather(*tasks)
    return results.count(True) >= (vote // 2 + 1)


async def achoice(backend:Backend, ctx:Context, prompt:str, choices:list[str], *, vote:int=3, positional:bool=True, **kwargs) -> str:
    """
    Answers a multiple-choice question. Returns an element of choices.

    Args:
        vote: When >=1, it samples multiple selections in each turn, and perform a majority voting.
        positional: Shuffle the order to present the elements to the LLM in order to mitigate the positional bias.

    """

    # note: constraint decoding does not respect pydantic.conint
    L = len(choices)
    class Choice(BaseModel):
        answer : Literal[*[ str(i) for i in range(L)]]

    async def choose(choices:list[str]) -> str:
        output, _ = await ainstruct(f"{prompt}\n" +
                                    f"Answer the index (0-{L-1}) of one of the following choices: \n" +
                                    "\n".join([f"index {i}: {c}" for i, c in enumerate(choices)]),
                                    ctx, backend,
                                    format=Choice, **kwargs)
        index = int(Choice.model_validate_json(output.value).answer)
        return choices[index]

    if positional:
        # enumerate random permutations while avoiding duplicaes
        shuffled = set()
        while len(shuffled) < vote:
            _choices = choices.copy()
            random.shuffle(_choices)
            shuffled.add(tuple(_choices))
        inputs = list(shuffled)
    else:
        inputs = [ choices for _ in range(vote) ]

    tasks = [choose(_choices) for _choices in inputs]

    chosen = await asyncio.gather(*tasks)

    counter = collections.Counter(chosen)

    return counter.most_common(1)[0][0]


class Core:
    """
    The Core powerup provides a core functionality for extracting the embedded reward model in the model.
    """
    pass

Core.abool = session_wrapper(abool)
Core.achoice = session_wrapper(achoice)
Core.bool = sync_wrapper(Core.abool)
Core.choice = sync_wrapper(Core.achoice)

