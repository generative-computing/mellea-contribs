
import random
import functools
import itertools
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






class Relation(Core):

    async def abinary(m:MelleaSession, criteria:str, x:str, y:str, vote:int=3,
                      symmetric:bool=False,
                      asymmetric:bool=False,
                      positional:bool=True,
                      shuffle:bool=True, **kwargs) -> bool:
        """Evaluates a query that evaluates a binary relation.

        Args:
            criteria: A natural language statement on variables X and Y. LLM decides if X and Y satisfy this critria, and this function returns yes if so.
            x: the first element
            y: the second element
            vote: an odd integer specifying the number of queries to make. The final result is a majority vote over the results. Since the LLM answers "yes"/"no", by default it counts "yes". If it is even, we add 1 to make it an odd number.
            symmetric: Declares the relation to be symmetric. Half of the queries swap x and y.
            asymmetric: Declares the relation to be asymmetric. Half of the queries swap x and y, and asks if they violate the criteria. This mitigates LLM's psycophancy bias toward answering "yes".
            positional: Half of the queries shuffle the order of presenting x and y. This mitigates the positional bias.
            shuffle: It shuffles the variation of queries (symmetric/positional variations).
                     This helps when you are making multiple queries with a small vote count (less than 2*2=4 variations).
                     For example, when shuffle = False and vote = 1, the query always contains the original x y in the x y order.
        Returns:
            bool.
        """

        assert not (symmetric and asymmetric), "symmetric and asymmetric flags are mutually exclusive"

        if vote % 2 == 0:
            FancyLogger.get_logger().warning(
                "the specified number of votes in a majority vote is even, making ties possible. Increasing the value by one to avoid this."
            )
            vote += 1

        if symmetric:
            if positional:
                prompts = [
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{x}\nY:{y}",
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{y}\nY:{x}",
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nY:{y}\nX:{x}",
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nY:{x}\nX:{y}",
                ]
            else:
                prompts = [
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{x}\nY:{y}",
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{y}\nY:{x}",
                ]
        elif asymmetric:
            if positional:
                prompts = [
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{x}\nY:{y}",
                    f"Do X and Y violate the following criteria? \nCriteria: {criteria}\nX:{y}\nY:{x}",
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nY:{y}\nX:{x}",
                    f"Do X and Y violate the following criteria? \nCriteria: {criteria}\nY:{x}\nX:{y}",
                ]
            else:
                prompts = [
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{x}\nY:{y}",
                    f"Do X and Y violate the following criteria? \nCriteria: {criteria}\nX:{y}\nY:{x}",
                ]
        else:
            if positional:
                prompts = [
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{x}\nY:{y}",
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nY:{y}\nX:{x}",
                ]
            else:
                prompts = [
                    f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{x}\nY:{y}",
                ]

        if shuffle:
            random.shuffle(prompts)

        tasks = [
            m.abool(p)
            for i, p in zip(range(vote),itertools.cycle(prompts))
        ]

        answers = asyncio.gather(*tasks)

        return answers.count(True) >= (vote // 2) + 1


    def binary(m:MelleaSession, criteria:str, x:str, y:str, vote:int=3,
               symmetric:bool=False,
               asymmetric:bool=False,
               positional:bool=True,
               shuffle:bool=True, **kwargs) -> bool:
        return _run_async_in_thread(abinary(m, criteria, x, y, symmetric, vote, **kwargs))



class Sorting(Arity):

