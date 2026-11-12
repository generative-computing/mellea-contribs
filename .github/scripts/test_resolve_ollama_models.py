"""Tests for resolve_ollama_models.py."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

# Ensure the script under test is importable when pytest is invoked from the
# repo root (the script lives next to this test file, not on sys.path).
sys.path.insert(0, str(Path(__file__).parent))

from resolve_ollama_models import (  # noqa: E402
    default_ollama_model,
    read_declared_models,
    required_models,
    validate_pulled,
)


class _FakeModelIdentifier:
    """Stand-in for mellea's ModelIdentifier carrying an ``ollama_name``."""

    def __init__(self, ollama_name: str | None) -> None:
        self.ollama_name = ollama_name


def _bind_default(default: Any):
    """Build a ``start_session`` stand-in whose ``model_id`` default is ``default``."""

    def start_session(model_id: Any = default) -> None: ...

    return start_session


# --- default_ollama_model ---------------------------------------------------


def test_resolves_ollama_name_from_model_identifier_default() -> None:
    start_session = _bind_default(_FakeModelIdentifier(ollama_name="granite4.2:3b"))
    assert default_ollama_model(start_session=start_session) == "granite4.2:3b"


def test_resolves_plain_string_default_verbatim() -> None:
    start_session = _bind_default("granite4.2:3b")
    assert default_ollama_model(start_session=start_session) == "granite4.2:3b"


def test_raises_when_default_has_no_ollama_name() -> None:
    start_session = _bind_default(_FakeModelIdentifier(ollama_name=None))
    with pytest.raises(RuntimeError, match="ollama"):
        default_ollama_model(start_session=start_session)


def test_raises_when_signature_has_no_model_id() -> None:
    def start_session() -> None: ...

    with pytest.raises(RuntimeError, match="model_id"):
        default_ollama_model(start_session=start_session)


# --- read_declared_models ---------------------------------------------------


def _write_pyproject(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(textwrap.dedent(body))
    return path


def test_reads_declared_models(tmp_path: Path) -> None:
    pyproject = _write_pyproject(
        tmp_path,
        """
        [tool.mellea-contribs.ci]
        ollama_models = ["granite3.3:8b", "qwen3:8b"]
        """,
    )
    assert read_declared_models(pyproject) == ["granite3.3:8b", "qwen3:8b"]


def test_declared_models_absent_returns_empty(tmp_path: Path) -> None:
    pyproject = _write_pyproject(
        tmp_path,
        """
        [tool.mellea-contribs.ci]
        skip_ollama = false
        """,
    )
    assert read_declared_models(pyproject) == []


def test_declared_models_no_ci_table_returns_empty(tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path, '[project]\nname = "x"\n')
    assert read_declared_models(pyproject) == []


def test_declared_models_rejects_non_list(tmp_path: Path) -> None:
    pyproject = _write_pyproject(
        tmp_path,
        """
        [tool.mellea-contribs.ci]
        ollama_models = "granite3.3:8b"
        """,
    )
    with pytest.raises(RuntimeError, match="ollama_models"):
        read_declared_models(pyproject)


# --- required_models --------------------------------------------------------


def test_required_models_unions_default_and_declared(tmp_path: Path) -> None:
    pyproject = _write_pyproject(
        tmp_path,
        """
        [tool.mellea-contribs.ci]
        ollama_models = ["granite3.3:8b", "qwen3:8b"]
        """,
    )
    start_session = _bind_default(_FakeModelIdentifier(ollama_name="granite4.2:3b"))
    assert required_models(pyproject, start_session=start_session) == [
        "granite4.2:3b",
        "granite3.3:8b",
        "qwen3:8b",
    ]


def test_required_models_default_only(tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path, '[project]\nname = "x"\n')
    start_session = _bind_default(_FakeModelIdentifier(ollama_name="granite4.2:3b"))
    assert required_models(pyproject, start_session=start_session) == ["granite4.2:3b"]


def test_required_models_dedupes_declared_default(tmp_path: Path) -> None:
    pyproject = _write_pyproject(
        tmp_path,
        """
        [tool.mellea-contribs.ci]
        ollama_models = ["granite4.2:3b", "qwen3:8b"]
        """,
    )
    start_session = _bind_default(_FakeModelIdentifier(ollama_name="granite4.2:3b"))
    assert required_models(pyproject, start_session=start_session) == [
        "granite4.2:3b",
        "qwen3:8b",
    ]


# --- validate_pulled --------------------------------------------------------


def test_validate_pulled_exact_match_passes() -> None:
    validate_pulled(["granite4.2:3b"], ["granite4.2:3b"])


def test_validate_pulled_tolerates_latest_tag() -> None:
    # `ollama list` prints `foo:latest` for a bare `foo` pull.
    validate_pulled(["granite4.2:3b", "mymodel"], ["granite4.2:3b", "mymodel:latest"])


def test_validate_pulled_raises_on_extra() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        validate_pulled(["granite4.2:3b"], ["granite4.2:3b", "llama3:8b"])


def test_validate_pulled_raises_on_missing() -> None:
    with pytest.raises(RuntimeError, match="missing"):
        validate_pulled(["granite4.2:3b", "qwen3:8b"], ["granite4.2:3b"])
