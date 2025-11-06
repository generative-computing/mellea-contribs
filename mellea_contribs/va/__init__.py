
import random
import functools
import itertools
import asyncio
from mellea import MelleaSession
from mellea.helpers.fancy_logger import FancyLogger
from mellea.helpers.event_loop_helper import _run_async_in_thread

from pydantic import BaseModel

from typing import Literal


def sync_wrapper(async_fn):
    """Wrap an async function so it can be called synchronously."""
    @functools.wraps(async_fn)
    def wrapper(*args, **kwargs):
        return _run_async_in_thread(async_fn(*args, **kwargs))
    return wrapper


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

    pass

Core.bool = sync_wrapper(Core.abool)
Core.choice = sync_wrapper(Core.achoice)


class Relation(Core):

    async def abinary(m:MelleaSession, criteria:str, x:str, y:str, *,
                      vote:int=3,
                      symmetric:bool=False,
                      asymmetric:bool=False,
                      reflexive:bool=False,
                      irreflexive:bool=False,
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
            reflexive: Declares the relation to be reflexive, i.e., if x == y, returns True immediately.
            irreflexive: Declares the relation to be irreflexive, i.e., if x == y, returns False immediately.
            positional: Permute the order of presenting x and y. This mitigates the positional bias.
            shuffle: It shuffles the variation of queries (symmetric/positional variations).
                     This helps when you are making multiple queries with a small vote count (less than 2*2=4 variations).
                     For example, when shuffle = False and vote = 1, the query always contains the original x y in the x y order.
        Returns:
            bool.
        """

        assert not (symmetric and asymmetric), "symmetric and asymmetric flags are mutually exclusive"

        if x == y:
            if reflexive:
                return True
            if irreflexive:
                return False

        if vote % 2 == 0:
            FancyLogger.get_logger().warning(
                "the specified number of votes in a majority vote is even, making ties possible. Increasing the value by one to avoid this."
            )
            vote += 1

        if symmetric:
            args = [(x,y),(y,x)]
            target = [True,True]
        elif asymmetric:
            args = [(x,y),(y,x)]
            target = [True,False]
        else:
            args = [(x,y)]
            target = [True]

        prompts = []
        for (x, y), t in zip(args, target):
            prompts.append((f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nX:{x}\nY:{y}", t))
            if positional:
                prompts.append((f"Do X and Y satisfy the following criteria? \nCriteria: {criteria}\nY:{y}\nX:{x}", t))

        if shuffle:
            random.shuffle(prompts)

        tasks = [
            m.abool(p)
            for i, (p, t) in zip(range(vote),itertools.cycle(prompts))
        ]

        answers = asyncio.gather(*tasks)

        answers = [ t == a for (p, t), a in zip(itertools.cycle(prompts), answers) ]

        return answers.count(True) >= (vote // 2) + 1


    async def aternary(m:MelleaSession, criteria:str, x:str, y:str, z:str, *,
                       vote:int=3,
                       symmetric:bool=False,
                       asymmetric:bool=False,
                       positional:bool=True,
                       shuffle:bool=True,
                       **kwargs) -> bool:
        """Evaluates a query that evaluates a ternary relation.

        Args:
            criteria: A natural language statement on variables X and Y. LLM decides if X and Y satisfy this critria, and this function returns yes if so.
            x: the first element
            y: the second element
            z: the third element
            vote: an odd integer specifying the number of queries to make. The final result is a majority vote over the results. Since the LLM answers "yes"/"no", by default it counts "yes". If it is even, we add 1 to make it an odd number.
            symmetric: Declares the relation to be symmetric wrto x and y. Half of the queries swap x and y.
            asymmetric: Declares the relation to be asymmetric wrto x and y. Half of the queries swap x and y, and asks if they violate the criteria. This mitigates LLM's psycophancy bias toward answering "yes".
            positional: The queries permutes the order of presenting x, y, z. This mitigates the positional bias.
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
            args = [(x,y,z),(y,x,z)]
            target = [True,True]
        elif asymmetric:
            args = [(x,y,z),(y,x,z)]
            target = [True,False]
        else:
            args = [(x,y,z)]
            target = [True]

        prompts = []
        for (x, y, z), t in zip(args, target):
            parts = [f"X:{x}", f"Y:{y}", f"Z:{z}"]
            if positional:
                for _parts in itertools.permutations(parts):
                    prompts.append(("\n".join([f"Do X, Y and Z satisfy the following criteria?", f"Criteria: {criteria}", *_parts]), t))
            else:
                prompts.append(("\n".join([f"Do X, Y and Z satisfy the following criteria?", f"Criteria: {criteria}", *parts]), t))

        if shuffle:
            random.shuffle(prompts)

        tasks = [
            m.abool(p)
            for i, (p, t) in zip(range(vote),itertools.cycle(prompts))
        ]

        answers = asyncio.gather(*tasks)

        answers = [ t == a for (p, t), a in zip(itertools.cycle(prompts), answers) ]

        return answers.count(True) >= (vote // 2) + 1


Relation.binary = sync_wrapper(Relation.abinary)
Relation.ternary = sync_wrapper(Relation.aternary)


