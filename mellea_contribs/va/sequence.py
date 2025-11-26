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
from .relation import Relation


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
    """
    Sequence powerup provides a set of sequence operations, such as
    mapping a list of strings,
    sorting a list of strings,
    selecting an element, or extracting the median according to some criteria.
    """

    async def amap(m:MelleaSession, variable:str, output:str, elems:list[str], **kwargs) -> list[str]:

        tasks = [
            m.ainstruct(f"Given a value of {variable}, answer the value of the output. \n"+
                        f"{variable}: {elem}\n" +
                        f"Output: {output}\n")
            for elem in elems
        ]

        return [o.value for o in asyncio.gather(*tasks)]

    async def afind(m:MelleaSession, variable:str, criteria:str, elems:list[str], **kwargs) -> str | None:

        """
        Returns any element which satisfies the criteria about the variable.
        It checks the criteria over the elements concurrently and returns the earliest element that satisfied the criteria,
        cancelling all running or pending LLM calls.

        Args:
            vote: When >=1, it samples multiple selections in each turn, and perform a majority voting.
        """

        if vote % 2 == 0:
            logger.warning(
                "the specified number of votes in a majority vote is even, making ties possible. Increasing the value by one to avoid this."
            )
            vote += 1

        async def per_elem(elem):
            tasks = [
                m.abool(f"Does {variable} satisfy the criteria?\n"+
                        f"{variable}: {elem}\n"+
                        f"Criteria: {criteria}")
                for _ in range(vote)
            ]
            return asyncio.gather(*tasks).count(True) >= (vote // 2 + 1), elem

        tasks = [
            per_elem(elem)
            for elem in elems
        ]

        async for answer, elem in asyncio.as_completed(*tasks):
            if answer:
                return elem
        pass

    async def amerge(m:MelleaSession, criteria:str, elems1:list[str], elems2:list[str], *,
                     vote:int=3,
                     positional:bool=True,
                     shuffle:bool=True, **kwargs) -> list[str]:
        """
        Given two lists already sorted according to the criteria,
        merge them into a list so that the resulting list is also sorted according to the criteria.
        """

        async def acmp(x, y):
            return await m.agt(criteria, x, y, vote=vote, positional=positional, shuffle=shuffle, **kwargs)

        return async_merge(elems1, elems2, acmp)

    async def asort(m:MelleaSession, criteria:str, elems:list[str], *,
                    vote:int=3,
                    positional:bool=True,
                    shuffle:bool=True, **kwargs) -> list[str]:

        async def acmp(x, y):
            return await m.agt(criteria, x, y, vote=vote, positional=positional, shuffle=shuffle, **kwargs)

        return async_merge_sort(elems, acmp)

    async def amax(m:MelleaSession, criteria:str, elems:list[str], *,
                   vote:int=3,
                   positional:bool=True,
                   shuffle:bool=True, **kwargs) -> str:

        async def acmp(x, y):
            return await m.agt(criteria, x, y, vote=vote, positional=positional, shuffle=shuffle, **kwargs)

        return async_max(elems, acmp)

    async def amedian(m:MelleaSession, criteria:str, elems:list[str], *,
                      exact = False,
                      vote:int=3,
                      positional:bool=True,
                      shuffle:bool=True,
                      block_size:int=5,
                      **kwargs) -> str:
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



Sequence.map = sync_wrapper(Sequence.amap)
Sequence.find = sync_wrapper(Sequence.afind)
Sequence.merge = sync_wrapper(Sequence.amerge)
Sequence.sort = sync_wrapper(Sequence.asort)
Sequence.max = sync_wrapper(Sequence.amax)
Sequence.median = sync_wrapper(Sequence.amedian)

