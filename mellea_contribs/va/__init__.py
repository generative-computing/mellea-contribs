
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

    async def agt(m:MelleaSession, criteria:str, x:str, y:str, *,
                  vote:int=3,
                  positional:bool=True,
                  shuffle:bool=True, **kwargs) -> bool:
        """Evaluates a query that evaluates a "greater-than" relation.

        Args:
            criteria: A natural language statement on variables X and Y. LLM decides if X and Y satisfy this critria, and this function returns yes if so.
            x: the first element
            y: the second element
            vote: an odd integer specifying the number of queries to make. The final result is a majority vote over the results. Since the LLM answers "yes"/"no", by default it counts "yes". If it is even, we add 1 to make it an odd number.
            positional: Permute the order of presenting x and y. This mitigates the positional bias.
            shuffle: It shuffles the variation of queries (symmetric/positional variations).
                     This helps when you are making multiple queries with a small vote count (less than 2*2=4 variations).
                     For example, when shuffle = False and vote = 1, the query always contains the original x y in the x y order.
        Returns:
            bool.
        """
        return await m.abinary(criteria, x, y,
                               vote=vote,
                               symmetric=False,
                               asymmetric=True,
                               reflexive=False,
                               irreflexive=True,
                               shuffle=shuffle, **kwargs)

    async def age(m:MelleaSession, criteria:str, x:str, y:str, *,
                  vote:int=3,
                  positional:bool=True,
                  shuffle:bool=True, **kwargs) -> bool:
        """Evaluates a query that evaluates a "greater-than-equal" relation.

        Args:
            criteria: A natural language statement on variables X and Y. LLM decides if X and Y satisfy this critria, and this function returns yes if so.
            x: the first element
            y: the second element
            vote: an odd integer specifying the number of queries to make. The final result is a majority vote over the results. Since the LLM answers "yes"/"no", by default it counts "yes". If it is even, we add 1 to make it an odd number.
            positional: Permute the order of presenting x and y. This mitigates the positional bias.
            shuffle: It shuffles the variation of queries (symmetric/positional variations).
                     This helps when you are making multiple queries with a small vote count (less than 2*2=4 variations).
                     For example, when shuffle = False and vote = 1, the query always contains the original x y in the x y order.
        Returns:
            bool.
        """
        return await m.abinary(criteria, x, y,
                               vote=vote,
                               symmetric=False,
                               asymmetric=True,
                               reflexive=True,
                               irreflexive=False,
                               shuffle=shuffle, **kwargs)

    async def aeq(m:MelleaSession, criteria:str, x:str, y:str, *,
                  vote:int=3,
                  positional:bool=True,
                  shuffle:bool=True, **kwargs) -> bool:
        """Evaluates a query that evaluates an equivalence relation.

        Args:
            criteria: A natural language statement on variables X and Y. LLM decides if X and Y satisfy this critria, and this function returns yes if so.
            x: the first element
            y: the second element
            vote: an odd integer specifying the number of queries to make. The final result is a majority vote over the results. Since the LLM answers "yes"/"no", by default it counts "yes". If it is even, we add 1 to make it an odd number.
            positional: Permute the order of presenting x and y. This mitigates the positional bias.
            shuffle: It shuffles the variation of queries (symmetric/positional variations).
                     This helps when you are making multiple queries with a small vote count (less than 2*2=4 variations).
                     For example, when shuffle = False and vote = 1, the query always contains the original x y in the x y order.
        Returns:
            bool.
        """
        return await m.abinary(criteria, x, y,
                               vote=vote,
                               symmetric=True,
                               asymmetric=False,
                               reflexive=True,
                               irreflexive=False,
                               shuffle=shuffle, **kwargs)


Relation.binary = sync_wrapper(Relation.abinary)
Relation.ternary = sync_wrapper(Relation.aternary)
Relation.gt = sync_wrapper(Relation.agt)
Relation.ge = sync_wrapper(Relation.age)
Relation.eq = sync_wrapper(Relation.aeq)


async def async_merge_sort(lst:list[str], acmp):
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    left = await async_merge_sort(lst[:mid], acmp)
    right = await async_merge_sort(lst[mid:], acmp)
    return await async_merge(left, right, acmp)

async def async_merge(left:list[str], right:list[str], acmp):
    result = []
    while left and right:
        if await acmp(left[0], right[0]):
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    return result + left + right

async def async_max(lst:list[str], acmp):
    if len(lst) <= 1:
        return lst[0]
    mid = len(lst) // 2
    left = await async_max(lst[:mid], acmp)
    right = await async_max(lst[mid:], acmp)
    if await acmp(left, right):
        return left
    else:
        return right

async def async_mom(seq:list[str], acmp, asort, block_size=5):
    """
    Median of medians algorithm for finding an approximate median. Worst-case runtime O(n)
    """

    async def median_fixed(seq):
        return await asort(seq)[len(seq)//2]

    if len(seq) <= block_size:
        return await median_fixed(seq)

    # Step 1: Divide into groups of block_size
    groups = itertools.batched(seq, block_size)

    # Step 2: Find median of each group
    medians = asyncio.gather(*[median_fixed(g) for g in groups])

    # Step 3: Recursively find the pivot
    return await async_mom(medians, acmp, asort, block_size=block_size)

async def async_quickselect(seq:list[str], k, acmp, asort, block_size=5):
    """
    Quickselect algorithm that uses median-of-medians for pivot selection. Worst-case runtime O(n^2)
    """

    pivot = await async_mom(medians, acmp, asort, block_size=block_size)

    # Step 4: Partition
    lows, highs = [], []
    for x in seq:
        if await acmp(x, pivot):
            lows.append(x)
        else:
            highs.append(x)

    # Step block_size: Recurse
    if k < len(lows):
        return await async_quickselect(lows, k, acmp, asort, block_size=block_size)
    elif k == len(lows):
        return pivot
    else:
        return await async_quickselect(highs, k - len(lows), acmp, asort, block_size=block_size)


class Sequence(Relation):

    async def asort(m:MelleaSession, criteria:str, elems:list[str], *,
                    vote:int=3,
                    positional:bool=True,
                    shuffle:bool=True, **kwargs) -> bool:

        async def acmp(x, y):
            return await m.agt(criteria, x, y, vote=vote, positional=positional, shuffle=shuffle, **kwargs)

        return async_merge_sort(elems, acmp)

    async def amax(m:MelleaSession, criteria:str, elems:list[str], *,
                   vote:int=3,
                   positional:bool=True,
                   shuffle:bool=True, **kwargs) -> bool:

        async def acmp(x, y):
            return await m.agt(criteria, x, y, vote=vote, positional=positional, shuffle=shuffle, **kwargs)

        return async_max(elems, acmp)

    async def amedian(m:MelleaSession, criteria:str, elems:list[str], *,
                      exact = False,
                      vote:int=3,
                      positional:bool=True,
                      shuffle:bool=True,
                      block_size:int=5,
                      **kwargs) -> bool:
        """
        If exact = True, use quickselect.
        Otherwise, return the approximate median returned by median of medians.
        """

        async def acmp(x, y):
            return await m.agt(criteria, x, y, vote=vote, positional=positional, shuffle=shuffle, **kwargs)

        async def asort(elems:list[str]):
            return await m.asort(criteria, x, y, vote=vote, positional=positional, shuffle=shuffle, **kwargs)

        if exact:
            return await async_quickselect(elems, len(elems)//2, acmp, asort, block_size=block_size)
        else:
            return await async_mom(elems, acmp, asort, block_size=block_size)



Sort.sort = sync_wrapper(Sort.asort)
Sort.max = sync_wrapper(Sort.amax)
Sort.median = sync_wrapper(Sort.amedian)


