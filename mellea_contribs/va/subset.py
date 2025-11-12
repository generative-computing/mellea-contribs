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
from .core import Core

class Subset(Core):
    """
    Subset powerup provides methods for selecting a subset of the input set.
    """

    async def afilter(m:MelleaSession,
                      criteria: str,
                      elems:list[str],
                      *,
                      vote:int=3,
                      **kwargs) -> list[str]:
        """
        Returns a subset whose elements all satisfy the criteria.

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
                m.abool("Does the input satisfy the criteria?\n"+
                        f"Criteria: {criteria}\n"+
                        f"Input: {elem}")
                for _ in range(vote)
            ]
            return asyncio.gather(*tasks).count(True) >= (vote // 2 + 1)

        tasks = [
            per_elem(elem)
            for elem in elems
        ]

        results = []
        for answer, elem in zip(asyncio.gather(*tasks), elems):
            if answer:
                results.append(elem)

        return results


    async def asubset(m:MelleaSession,
                      description:str,
                      criteria: str,
                      elems:list[str],
                      k:int,
                      *,
                      vote:int=3,
                      positional:bool=True,
                      **kwargs) -> list[str]:
        """
        Greedily select a k-elements subset from elems, maximizing the given criteria.

        Args:
            description: A decription of what the current and the output subset is meant to represent.
            criteria: A decription of the desired property of the returned subset.
            elems: The universe to select the subset from.
            k: The number of elements to select from elems.
            vote: When >=1, it samples multiple selections in each turn, and perform a majority voting.
            positional: Shuffle the order to present the elements to the LLM in order to mitigate the positional bias.

        The criteria is assumed to be contain a modular or submodular aspect.


        Example 1:

        description = "We are building a team of culturally diverse members."

        criteria = "Maximize the cultural diversity among the members."


        Example 2:

        description = ("We need set of past legal cases that helps defending our case. "
                       "In our case, the defandant has ..."
                       "We want to see a variety of cases that are relevant to ours but"
                       "are also different from each other.")

        criteria = "Minimize the ovelap with the documents in the current set while staying relevant to our case."
        """

        current = []
        remaining = elems.copy()

        for _ in range(k):
            chosen = await m.achoice(f"{description}\n"
                                     "Choose the best element to add to the current set following the criteria:\n"
                                     f"Criteria: {criteria}\n" +
                                     "Current set:\n" +
                                     "\n".join(current) + "\n",
                                     remaining,
                                     vote=vote,
                                     positional=positional,
                                     **kwargs)
            current.append(chosen)
            remaining.remove(chosen)

        return current


Subset.filter = sync_wrapper(Subset.afilter)
Subset.subset = sync_wrapper(Subset.asubset)
