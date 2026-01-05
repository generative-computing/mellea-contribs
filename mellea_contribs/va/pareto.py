import random
import functools
import itertools
import asyncio
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mellea import MelleaSession
from mellea.backends import Backend
from mellea.stdlib.base import Context
from mellea.stdlib.functional import ainstruct
from mellea.helpers.fancy_logger import FancyLogger
from mellea.helpers.event_loop_helper import _run_async_in_thread

from pydantic import BaseModel

from typing import (
    Literal,
    Callable,
    TypeVar,
    List,
    Iterable,
    Awaitable,
)

import numpy as np

from .util import session_wrapper, sync_wrapper
from .sequence import Sequence, asort
from .relation import agt

T = TypeVar("T")

async def async_all(awaitables: Iterable[Awaitable[bool]]) -> bool:
    """
    Asynchronously evaluate awaitables like built-in all().

    - Returns False immediately when any awaitable resolves to False
    - Cancels all remaining tasks when returning early
    - Returns True only if all awaitables resolve to True
    """
    tasks = {asyncio.create_task(aw) for aw in awaitables}

    try:
        for completed in asyncio.as_completed(tasks):
            result = await completed
            if not result:
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Ensure cancellation is propagated
                await asyncio.gather(*tasks, return_exceptions=True)
                return False

        return True

    finally:
        # Safety net: cancel anything still pending
        for task in tasks:
            if not task.done():
                task.cancel()

async def async_any(awaitables: Iterable[Awaitable[bool]]) -> bool:
    """
    Asynchronously evaluate awaitables like built-in any().

    - Returns True immediately when any awaitable resolves to True
    - Cancels all remaining tasks when returning early
    - Returns False only if all awaitables resolve to False
    """
    tasks = {asyncio.create_task(aw) for aw in awaitables}

    try:
        for completed in asyncio.as_completed(tasks):
            result = await completed
            if result:
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Ensure cancellation is propagated
                await asyncio.gather(*tasks, return_exceptions=True)
                return True

        return False

    finally:
        # Safety net: cancel anything still pending
        for task in tasks:
            if not task.done():
                task.cancel()

async def async_kung(criteria:list[str], elems:list[str], acmp, asort) -> list[str]:
    """
    Kung's divide-and-conquer pareto-front computation algorithm, runtime n log^{d-2} n.
    """

    assert len(criteria) > 0

    if len(criteria) == 1:
        return await asort(criteria[0], elems)

    elems = await asort(criteria[0], elems)

    half = len(elems)//2
    elems1, elems2 = elems[:half], elems[half:]

    front1 = await async_kung(criteria[1:], elems1, acmp, asort)

    async def dominated(elem):
        return await \
            async_any([
                async_all([
                    acmp(criterion, elem, front_elem)
                    for criterion in criteria[1:]
                ])
                for front_elem in front1
            ])

    elems2_pruned = []
    for elem2 in elems2:
        if not await dominated(elem2):
            elems2_pruned.append(elem2)

    front2 = await async_kung(criteria[1:], elems2_pruned, acmp, asort)

    return front1 + front2


async def apareto(backend:Backend, ctx:Context,
                  criteria:list[str],
                  elems:list[str],
                  vote:int=3,
                  positional:bool=True,
                  shuffle:bool=True,
                  **kwargs) -> list[str]:
    """Compute the pareto front of elements based on multiple comparison criteria.

    The algorithm follows the divide-and-conquer algorithm in [1].

    Args:
        criteria: a list of natural language comparison criteria between X and Y.
        elems: list of strings to compute the pareto front for.

        **kwargs: accepts vote, positional, shuffle.

    Returns:
        The pareto front as list[str]

    [1] Kung, Luccio, and Preparata "On Finding the Maxima of a Set of Vectors",
    Journal of the ACM (JACM) 22.4 (1975): 469-476.

    """

    async def acmp(criterion, x, y):
        return await agt(backend, ctx, criterion, x, y, vote=vote, positional=positional, shuffle=shuffle, **kwargs)

    async def asort(criterion, elems:list[str]):
        return await asort(backend, ctx, criterion, elems, vote=vote, positional=positional, shuffle=shuffle, **kwargs)

    return await async_kung(criteria, elems, acmp, asort)


class Pareto(Sequence):
    pass

Pareto.apareto = session_wrapper(apareto)
Pareto.pareto = sync_wrapper(Pareto.apareto)

