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
    async def asubset(m:MelleaSession,
                      description:str,
                      criteria: str,
                      elems:list[str],
                      k:int,
                      *,
                      vote:int=3,
                      positional:bool=True,
                      **kwargs):
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


Subset.subset = sync_wrapper(Subset.asubset)
