"""Smoke test for mellea-contribs-acp."""

import importlib


def test_import() -> None:
    """The subpackage imports cleanly and exposes ACPBackend."""
    module = importlib.import_module("mellea_contribs.acp")
    assert hasattr(module, "ACPBackend")


def test_acpbackend_instantiates() -> None:
    """ACPBackend can be instantiated without starting a subprocess."""
    from mellea_contribs.acp import ACPBackend

    backend = ACPBackend()
    assert backend._model_id == "bob"
    assert backend._provider == "acp"
    assert backend._session_id is None


def test_acpbackend_custom_command() -> None:
    """ACPBackend respects a custom command."""
    from mellea_contribs.acp import ACPBackend

    backend = ACPBackend(command=["vibe-acp"])
    assert backend._model_id == "vibe-acp"
    assert backend._command == ["vibe-acp"]
