import random
import functools
import itertools
import asyncio
from mellea import MelleaSession
from mellea.helpers.fancy_logger import FancyLogger
from mellea.helpers.event_loop_helper import _run_async_in_thread

from pydantic import BaseModel

from typing import Literal

from .util import sync_wrapper

class YesNo(BaseModel):
    answer : Literal["yes","no"]

class Core:

    async def abool(m:MelleaSession, prompt:str, **kwargs) -> bool:

        output = await m.ainstruct(f"{prompt} Answer yes or no.",
                                   format=YesNo, **kwargs)

        yesno = YesNo.model_validate_json(output.value)

        return yesno.answer == "yes"

    async def achoice(self:MelleaSession, prompt:str, choices:list[str], *, vote:int=3, positional:bool=True, **kwargs) -> str:

        # note: constraint decoding does not respect pydantic.conint
        L = len(choices)
        class Choice(BaseModel):
            answer : Literal[*[ str(i) for i in range(L)]]

        async def choose(choices:list[str]) -> str:
            output = await self.ainstruct(f"{prompt}\n" +
                                          f"Answer the index (0-{L-1}) of one of the following choices: \n" +
                                          "\n".join([f"index {i}: {c}" for i, c in enumerate(_choices)]),
                                          format=Choice, **kwargs)
            index = int(Choice.model_validate_json(output.value))
            return choices[index]

        if positional:
            # enumerate random permutations while avoiding duplicaes
            shuffled = set()
            while len(shuffled) < vote:
                _choices = choices.copy()
                random.shuffle(_choices)
                shuffled.add(tuple(choices))
            inputs = list(shuffled)
        else:
            inputs = [ choices for _ in range(vote) ]

        tasks = [choose(_choices) for _choices in inputs]

        choices = asyncio.gather(*tasks)

        counter = Counter(choices)

        return counter.most_common(1)[0][0]

    pass

Core.bool = sync_wrapper(Core.abool)
Core.choice = sync_wrapper(Core.achoice)

