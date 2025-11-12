
import random
import functools
import itertools
import asyncio
import dataclasses
from mellea import MelleaSession
from mellea.helpers.fancy_logger import FancyLogger
from mellea.helpers.event_loop_helper import _run_async_in_thread

from pydantic import BaseModel

from typing import Literal

import numpy as np

from sklearn.base import ClusterMixin

logger = FancyLogger.get_logger()

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
            logger.warning(
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
            logger.warning(
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



Sort.sort = sync_wrapper(Sort.asort)
Sort.max = sync_wrapper(Sort.amax)
Sort.median = sync_wrapper(Sort.amedian)



@dataclasses.dataclass
class Triplet:
    z : str                     # anchor
    x : str
    y : str
    z_index : int               # z's index in items
    x_index : int               # x's index in items
    y_index : int               # y's index in items

    def swap(self):
        "swaps x and y"
        return Triplet(self.z, self.y, self.x, self.z_index, self.y_index, self.x_index)


def sample_triplets(items: list[str],
                    triplets_per_item: int | None = None,
                    num_triplets:      int | None = None,
                    repeat_x: int = 1,
                    ) -> list[Triplet]:
    """Randomly sample a list of triplet comparison queries.

    Args:
      items               : input
      triplets_per_item   : how many triplets to generate relative to the number of items.
      num_triplets        : the number of triplets to generate.
      repeat_x            : how many times we reuse the same z, x for sampling y.

    Either triplets_per_item or num_triplets must be specified.
    triplets_per_item and num_triplets are mutually exclusive (cannot be specified at the same time).

    """
    N = len(items)

    assert (num_triplets is not None) or (triplets_per_item is not None), \
        "Specify either num_triplets and triplets_per_item."
    assert (num_triplets is None) or (triplets_per_item is None), \
        "num_triplets and triplets_per_item are mutually exclusive; do not specify both."
    if num_triplets is None:
        assert isinstance(triplets_per_item, int)
        logger.info(f"num_triplets = triplets_per_item * len(items) = {triplets_per_item} * {N} = {triplets_per_item * N}")
        num_triplets = triplets_per_item * N

    # make sure z covers all elements
    assert num_triplets / N >= 1, \
        ("Some items are never used as an anchor z. Increase num_triplets or triplets_per_item: "
         f"num_triplets / len(items) = {num_triplets} / {N} = {num_triplets / N}")

    # make sure z covers all elements even if we sample multiple triplets with the same x
    if repeat_x > num_triplets / N:
        logger.warning(f"Some items are never used as an anchor z because of too large repeat_x. Overriding it with {num_triplets / N}: "
                       f"repeat_x = {repeat_x} > "
                       f"num_triplets / len(items) = {num_triplets} / {N} = {num_triplets / N}.")
        repeat_x = num_triplets // N

    # switch to the exhaustive mode if num_triplets is large enough
    all_triplets = N * (N-1) * (N-2)
    logger.info(f"all_triplets = N * (N-1) * (N-2) = {all_triplets}, where N = {N}")
    if num_triplets > all_triplets:
        logger.warning(f"num_triplets = {num_triplets} is large enough to enumerate all triplets (> {N} * {(N-1)} * {(N-2)} = {all_triplets}). "
                       f"Switching to the exhaustive mode.")
        exhaustive = True
        num_triplets = all_triplets
    else:
        exhaustive = False

    triplets: list[Triplet] = []

    bar = tqdm(total=num_triplets, desc="sampling triplets")

    if exhaustive:
        for i, z in enumerate(items):
            for j, x in enumerate(items):
                if i == j:
                    continue
                for k, y in enumerate(items):
                    if k == i or k == j:
                        continue
                    triplets.append(Triplet(z, x, y, i, j, k))
                    bar.update()
        assert len(triplets) == all_triplets
        return triplets

    def sample_except(blacklist:set[str]):
        while True:
            sample = random.choice(items)
            if sample not in blacklist:
                return sample

    for z in cycle(items):
        x = sample_except({z})
        for _ in range(repeat_x):
            y = sample_except({z,x})
            triplets.append(Triplet(z, x, y, items.index(z), items.index(x), items.index(y)))
            bar.update()
            if len(triplets) >= num_triplets:
                return triplets

def update(embeddings: np.ndarray, triplets: list[Triplet], alpha: float, lr: float) -> int:
    """ Update embeddings using the t-STE gradient for each triplet. """
    violations_fixed: int = 0
    for idx, t in enumerate(triplets):
        xi = embeddings[t.z_index]
        xj = embeddings[t.x_index]
        xl = embeddings[t.y_index]

        # Squared distances
        dij = np.sum((xi - xj) ** 2)
        dil = np.sum((xi - xl) ** 2)

        # Student-t similarities
        sij = (1 + dij / alpha) ** (-(alpha + 1) / 2)
        sil = (1 + dil / alpha) ** (-(alpha + 1) / 2)
        pijl = sij / (sij + sil)

        # Gradients (see t-STE paper)
        grad_coeff = (alpha + 1) / alpha
        grad_xi = grad_coeff * (
            (1 - pijl) * (xj - xi) / (1 + dij / alpha)
            - (1 - pijl) * (xl - xi) / (1 + dil / alpha)
        )
        grad_xj = grad_coeff * (1 - pijl) * (xi - xj) / (1 + dij / alpha)
        grad_xl = -grad_coeff * (1 - pijl) * (xi - xl) / (1 + dil / alpha)

        # Update embeddings
        embeddings[t.z_index] = (xi + lr * grad_xi)
        embeddings[t.x_index] = (xj + lr * grad_xj)
        embeddings[t.y_index] = (xl + lr * grad_xl)
        violations_fixed += 1
    return violations_fixed

default_prompt = "Considering the nature of X, Y and Z, is X more similar to Z than Y is to Z? "

class Cluster(Relation):

    def query_triplets(m:MelleaSession, triplets: list[Triplet], prompt: str) -> list[Triplet]:
        """Given a triplet comparison query, perform the query using the LLM. """

        answers = asyncio.gather(*[m.achoice(prompt + f"\nZ: {t.z}\nX: {t.x}\nY: {t.y}\n" , ["X", "Y"])
                                   for t in triplets])

        logger.info(f"Queried {len(triplets)} triplets.")

        for idx, (t, a) in enumerate(zip(triplets, answers)):
            logger.debug(f"Triplet {idx+1}:  Z(anchor): {t.z}  X: {t.x}  Y: {t.y}  result: {a}")

        return [t if a == "X" else t.swap()
                for t, a in zip(triplets, answers) ]

    async def acluster(m:MelleaSession, criteria:str, elems:list[str],
                       model : ClusterMixin,
                       *,
                       vote:int=3,
                       positional:bool=True,
                       shuffle:bool=True,
                       #
                       ndims: int = 2,
                       lr: float = 0.020,
                       max_iterations: int = 100,
                       tolerance: float = 1e-4,
                       alpha: float | None = None,
                       num_triplets: int | None = None,
                       triplets_per_item: int | None = None,
                       repeat_x: int = 3,
                       #
                       **kwargs):
        """
        Generate Triplet Embeddings of the given strings, and run clustering

        Args:
            criteria: triplet comparison criteria
            elems: list of strings to cluster
            model: an instance of sklearn.base.ClusterMixin, such as sklearn.cluster.KMEANS, sklearn.cluster.AgglomerativeClustering

            vote: an odd integer specifying the number of queries to make. The final result is a majority vote over the results. Since the LLM answers "yes"/"no", by default it counts "yes". If it is even, we add 1 to make it an odd number.
            positional: Permute the order of presenting x and y. This mitigates the positional bias.
            shuffle: It shuffles the variation of queries (symmetric/positional variations).
                     This helps when you are making multiple queries with a small vote count (less than 2*2=4 variations).
                     For example, when shuffle = False and vote = 1, the query always contains the original x y in the x y order.

            ndims: number of dimensions for embeddings
            lr: weight to give each triplet when updating embeddings
            max_iterations: number of times to use LLM triplets to update embeddings
            tolerance: hyperparamater; ???
            alpha: hyperparameter; ???
            num_triplets: the number of triplets to generate.
            triplets_per_item: number of triplets to sample per item (will result in len(items) * triplets_per_item triplets)
            repeat_x: how many times we reuse the same z, x for sampling y.
            verbose: boolean to determine whether or not to provide verbose output
            clustering_method: clustering method

        Returns:
            Dictionary representing each label in items to its associated coordinate

        Either triplets_per_item or num_triplets must be specified.
        triplets_per_item and num_triplets are mutually exclusive (cannot be specified at the same time).
        """

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        start_time = datetime.now()
        # Set alpha default based on ndims if not provided
        if alpha is None:
            alpha = ndims - 1
        if criteria is None:
            criteria = default_criteria

        N: int = len(items)

        logger.info(f"Starting triplet embedding with N={N} items...")
        logger.info(f"Algorithm parameters:")
        logger.info(f"  Embedding dimensions (r): {ndims}")
        logger.info(f"  Learning rate: {lr}")
        logger.info(f"  Max iterations: {max_iterations}")
        logger.info(f"  Triplets per item: {triplets_per_item}")
        logger.info(f"  Reuse the same z and x for: {repeat_x} times")
        logger.info(f"  Tolerance: {tolerance}")
        logger.info(f"  Alpha (DoF): {alpha}")

        # Initialize random embeddings
        embeddings = np.random.normal(0, 0.1, (len(items), ndims))

        # Show initial positions
        logger.debug(f"Generated initial random embeddings in {ndims}D space")

        # Generate LLM triplets ONCE
        triplets = sample_triplets(items,
                                   triplets_per_item=triplets_per_item,
                                   num_triplets=num_triplets,
                                   repeat_x=repeat_x,)
        logger.debug(f"Using {len(triplets)} LLM-judged triplets for all iterations")

        # swap X/Y of triplets using LLM. Now X is always closer to anchor Z than Y is to anchor Z
        triplets = m.query_triplets(triplets, criteria)

        # Iterative improvement
        stat = {
            "violations_fixed": 0,
            "convergence_ratio": 0.0,
        }
        for iteration in tqdm(range(max_iterations), desc="updating the embedding (outer loop)", position=0, postfix=stat):
            # Use the same triplets every iteration
            violations_fixed: int = update(embeddings, triplets, alpha, lr)
            convergence_ratio: float = violations_fixed / len(triplets) if len(triplets) > 0 else 0

            stat["violations_fixed"] = violations_fixed
            stat["convergence_ratio"] = convergence_ratio

            if convergence_ratio < tolerance:
                logger.debug(f"Converged early at iteration {iteration + 1} (ratio < {tolerance})")
                break

        elapsed_time = datetime.now() - start_time
        formatted = str(elapsed_time).split('.')[0]
        logger.debug(f"Elapsed time: {formatted}")

        return model.fit_predict(embeddings)


Cluster.cluster = sync_wrapper(Sort.acluster)
