"""Author: IBM Research – Mellea Agent Team
Maintainer: Mellea Agent - IBM Research

Purpose: GroundingContextFormatter is a reusable component for constructing grounding context blocks for Mellea agents. It organizes arbitrary
context fields into a structured, LLM-friendly format.

What it Does:
- Accepts multiple context fields (logs, events, metrics, etc.).
- Automatically skips empty fields (None, "", [], {}, ()).
- Serializes lists and dictionaries with pretty-printing for readability.
- Returns either a plain string or a TemplateRepresentation object.

When should an agent use it?
- When the agent needs to provide structured grounding context to an LLM.
- When context contains multiple heterogeneous fields (logs, metrics, etc.).
- When the agent requires clean, human-readable formatting for debugging or interpretability.
"""

import json
from typing import Any

from mellea.stdlib.base import Component, TemplateRepresentation


class GroundingContextFormatter(Component):
    def __init__(
        self, return_template: bool = False, indent: int = 2, **context_fields: Any
    ):
        super().__init__()
        self.context_fields = context_fields
        self.return_template = return_template
        self.indent = indent

    def parts(self):
        return []

    def format_for_llm(self):
        non_empty = {
            k: v
            for k, v in self.context_fields.items()
            if v not in (None, "", [], {}, ())
        }

        if not non_empty:
            if self.return_template:
                return TemplateRepresentation(obj=self, args={})
            return ""

        blocks = []
        for name, value in non_empty.items():
            section_name = self._flatten_context_name(name)
            serialized = self._serialize(value)
            blocks.append(f"### {section_name}\n{serialized}")

        output = "\n\n".join(blocks)

        if self.return_template and not non_empty:
            return TemplateRepresentation(obj=self, template="", args={"context": ""})

        return output

    def _serialize(self, value: Any) -> str:
        try:
            if isinstance(value, (dict, list)):
                return json.dumps(value, indent=self.indent, ensure_ascii=False)
        except TypeError:
            return str(value)
        return str(value)

    def _flatten_context_name(self, name: str) -> str:
        return name
