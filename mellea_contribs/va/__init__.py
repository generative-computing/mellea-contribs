
import functools
import asyncio
from mellea import MelleaSession
from mellea.helpers.fancy_logger import FancyLogger
from mellea.helpers.event_loop_helper import _run_async_in_thread

from pydantic import BaseModel

from typing import Literal





class YesNo(BaseModel):
    answer : Literal["yes","no"]

class Core:

    async def abool(m:MelleaSession, prompt:str, **kwargs) -> bool:

        output = await m.ainstruct(f"{prompt} Answer yes or no.",
                                   format=YesNo, **kwargs)

        yesno = YesNo.model_validate_json(output.value)

        return yesno.answer == "yes"

    async def achoice(self:MelleaSession, prompt:str, choices:list[str], **kwargs) -> str:

        class Choice(BaseModel):
            answer : Literal[choices]

        output = await self.ainstruct(f"{prompt} Respond with one of the following answers: " + ",".join([f"'{c}'" for c in choices]) + ".",
                                      format=Choice, **kwargs)

        choice = Choice.model_validate_json(output.value)

        return choice.answer

    def bool(m:MelleaSession, prompt:str, **kwargs) -> bool:
        return _run_async_in_thread(abool(m, prompt, **kwargs))

    def choice(m:MelleaSession, prompt:str, choices:list[str], **kwargs) -> str:
        return _run_async_in_thread(achoice(m, prompt, **kwargs))






class Arity(Core):

    async def abinary(m:MelleaSession, criteria:str, x:str, y:str, symmetric:bool=True, vote:int=3, **kwargs) -> bool:
        """Evaluates a binary boolean function.
        """

        if vote % 2 == 0:
            FancyLogger.get_logger().warning(
                "the specified number of votes in a majority vote is even, making ties possible. Increasing the value by one to avoid this."
            )
            vote += 1

        if symmetric:
            tasks = [
                m.abool(f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{x}\nY:{y}")
                for _ in range(vote // 2 + 1)
            ] + [
                m.abool(f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{y}\nY:{x}")
                for _ in range(vote // 2)
            ]

        else:
            tasks = [
                m.abool(f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{x}\nY:{y}")
                for _ in range(vote)
            ]

        answers = asyncio.gather(*tasks)

        return (answers[:majority].count(True) + answers[majority:].count(False)) >= majority


    def binary(m:MelleaSession, criteria:str, x:str, y:str, symmetric:bool=True, vote:int=3, **kwargs) -> bool:
        return _run_async_in_thread(abinary(m, criteria, x, y, symmetric, vote, **kwargs))



class Sorting(Arity):

