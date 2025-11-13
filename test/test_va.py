import pytest

from mellea.backends.vllm import LocalVLLMBackend

from mellea_contribs.va import Core, Relation, Sequence, Subset, Cluster

from mellea import start_session
from mellea.stdlib.requirement import req
from mellea.stdlib.sampling import RejectionSamplingStrategy

import timer

MelleaSession.powerup(Core)
MelleaSession.powerup(Relation)
MelleaSession.powerup(Sequence)
MelleaSession.powerup(Subset)
MelleaSession.powerup(Cluster)

@pytest.fixture(scope="module")
def m() -> MelleaSession:
    return MelleaSession(backend=LocalVLLMBackend("Qwen/Qwen3-1.7B"))


def test_core(m: MelleaSession):
    assert m.bool("Is a number 2 even?")
    assert not m.bool("Is a number 5 even?")
    assert m.choice("Which country is in Asia?", ["United States", "Norway", "Japan", "France", "Namibia"]) == "Japan"

def test_relation(m: MelleaSession):
    assert m.gt("The population of X is larger than that of Y.", "China", "Singapore")

    t1 = time.now()
    answer = m.eq("People in country X and country Y speak the same language.", "Spain", "Mexico")
    t2 = time.now()
    assert answer

    t3 = time.now()
    answer = m.eq("People in country X and country Y speak the same language.", "Spain", "Spain")
    t4 = time.now()
    assert answer
    assert t4 - t3 < 1.0
    assert t4 - t3 < t2 - t1

def test_sequence(m: MelleaSession):

    assert m.map("X", "X+1", ["3", "5"]) == ["4", "6"]

    assert m.find("X", "X is a country in Asia.", ["United States", "Norway", "Japan", "France", "Namibia"]) == "Japan"

    messages = [
        # "You are the worst scum in this world",
        "I hate you",
        "I dislike you",
        # "You are a bit annoying",
        "You are okay",
        # "You are not bad",
        # "You are kind of nice",
        "I like you",
        "I love you",
        # "Oh my gosh you are the best person in the world"
    ]
    random.shuffle(messages)

    results = m.sort("Message X shows a more positive sentiment than message Y does.", messages)

    assert results.find("I love you") > results.find("I hate you")

    assert m.max("Message X shows a more positive sentiment than message Y does.", messages) == "I love you"

    assert m.median("Message X shows a more positive sentiment than message Y does.", messages) == "You are okay"


def test_subset(m: MelleaSession):

    assert m.filter("X", "X is an insect", ["crow", "dolphin", "cockroach", "cicada"]) == ["cockroach", "cicada"]

    subset = m.subset("We need a set of things with different colors.",
                      "Select an element whose color is different from any of the current set.",
                      ["crow", "orange", "tomato", "cucumber", "coal", "strawberry"])

    assert "orange" in subset
    assert ("tomato" in subset) != ("strawberry" in subset)
    assert ("crow" in subset) != ("coal" in subset)

