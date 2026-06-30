# mellea-contribs-{{ cookiecutter.dir_name }}

{{ cookiecutter.short_description }}

## Installation

```bash
pip install "mellea-contribs[{{ cookiecutter.dir_name }}] @ https://github.com/generative-computing/mellea-contribs/releases/latest/download/mellea_contribs-X.Y.Z-py3-none-any.whl"
```

(Update `X.Y.Z` to the latest release version. See the contribs README for details.)

## Layout convention

The source is laid out flat (`<core mirror>/...`); the wheel exposes the namespaced path:

- On disk:  `{{ cookiecutter.core_path | replace('.', '/') }}/{{ cookiecutter.name }}.py`
- Imports:  `from mellea_contribs.{{ cookiecutter.name }}.{{ cookiecutter.core_path }}.{{ cookiecutter.name }} import …`

The doubled segment (e.g., `mellea_contribs.{{ cookiecutter.name }}.{{ cookiecutter.core_path }}`) is deliberate — outer = subpackage; inner = identical to core.

## Development

```bash
cd {{ cookiecutter.dir_name }}
uv sync --all-extras
uv run pytest -m "not qualitative and not e2e"
```
