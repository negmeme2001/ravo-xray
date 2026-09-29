"""Versioned prompt and canned-message loading."""

from __future__ import annotations

import hashlib
import string
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROMPT_DIR = Path(__file__).parent
_LANGUAGES = frozenset({"ar", "en"})
_MESSAGE_SECTIONS = frozenset({"shopper", "model"})


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str
    sha8: str


@dataclass(frozen=True)
class _Catalog:
    prompts: dict[str, Prompt]
    variables: dict[str, frozenset[str]]
    messages: dict[str, dict[str, str]]
    versions: dict[str, str]


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def _read_markdown(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as file:
        text = file.read()
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith(("\n", "\r")):
        return text[:-1]
    return text


def _template_variables(text: str) -> frozenset[str]:
    fields = []
    for _, field_name, format_spec, _ in string.Formatter().parse(text):
        if field_name is not None:
            if not field_name:
                raise ValueError("positional template fields are not supported")
            fields.append(field_name)
        if format_spec:
            fields.extend(_template_variables(format_spec))
    return frozenset(fields)


def _version_entry(name: str, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    active = config.get("active")
    versions = config.get("versions")
    if not isinstance(active, str) or not isinstance(versions, dict) or active not in versions:
        raise ValueError(f"invalid active version for {name!r}")
    version_metadata = versions[active]
    if not isinstance(version_metadata, dict):
        raise ValueError(f"invalid version metadata for {name!r}@{active}")
    return active, version_metadata


def _validate_version_files(root: Path, name: str, config: dict[str, Any]) -> None:
    versions = config.get("versions")
    if not isinstance(versions, dict):
        raise ValueError(f"missing versions for {name!r}")
    for version, version_metadata in versions.items():
        if not isinstance(version, str) or not isinstance(version_metadata, dict):
            raise ValueError(f"invalid version metadata for {name!r}")
        file_name = version_metadata.get("file")
        note = version_metadata.get("note")
        if not isinstance(file_name, str) or not isinstance(note, str) or not note.strip():
            raise ValueError(f"incomplete version metadata for {name!r}@{version}")
        path = root / file_name
        if not path.is_file():
            raise FileNotFoundError(path)


def _load_messages(path: Path) -> dict[str, dict[str, str]]:
    document = _read_toml(path)
    if set(document) != _MESSAGE_SECTIONS:
        raise ValueError("messages must contain shopper and model sections")
    messages: dict[str, dict[str, str]] = {}
    for section_name, section in document.items():
        if not isinstance(section, dict):
            raise ValueError(f"invalid message section {section_name!r}")
        for key, translations in section.items():
            if key in messages or not isinstance(translations, dict) or not translations:
                raise ValueError(f"invalid message table {key!r}")
            unknown_languages = set(translations) - _LANGUAGES
            if unknown_languages:
                names = ", ".join(sorted(unknown_languages))
                raise ValueError(f"unknown language key for {key!r}: {names}")
            if not all(isinstance(message_text, str) for message_text in translations.values()):
                raise ValueError(f"message values for {key!r} must be strings")
            messages[key] = dict(translations)
    return messages


def _declared_variables(name: str, config: dict[str, Any]) -> frozenset[str]:
    declared = config.get("variables")
    if not isinstance(declared, list) or not all(
        isinstance(variable_name, str) for variable_name in declared
    ):
        raise ValueError(f"invalid template variables for {name!r}")
    return frozenset(declared)


def _version_texts(root: Path, config: dict[str, Any]) -> dict[str, str]:
    return {
        version: _read_markdown(root / version_metadata["file"])
        for version, version_metadata in config["versions"].items()
    }


def _validate_templates(
    name: str,
    version_texts: dict[str, str],
    expected: frozenset[str],
) -> None:
    for version, version_text in version_texts.items():
        found = _template_variables(version_text)
        if found != expected:
            raise ValueError(
                f"template variables for {name!r}@{version} are {sorted(found)}, "
                f"expected {sorted(expected)}"
            )


def _load_prompt(
    root: Path,
    name: str,
    config: dict[str, Any],
) -> tuple[Prompt, frozenset[str]]:
    _validate_version_files(root, name, config)
    active, _ = _version_entry(name, config)
    expected = _declared_variables(name, config)
    prompt_texts = _version_texts(root, config)
    _validate_templates(name, prompt_texts, expected)
    text = prompt_texts[active]
    prompt = Prompt(
        name=name,
        version=active,
        text=text,
        sha8=hashlib.sha256(text.encode("utf-8")).hexdigest()[:8],
    )
    return prompt, expected


def _load_prompts(
    root: Path,
    prompt_configs: dict[str, Any],
) -> tuple[dict[str, Prompt], dict[str, frozenset[str]], dict[str, str]]:
    prompts: dict[str, Prompt] = {}
    variables: dict[str, frozenset[str]] = {}
    versions: dict[str, str] = {}
    for name, config in prompt_configs.items():
        if not isinstance(name, str) or not isinstance(config, dict):
            raise ValueError("invalid prompt manifest entry")
        prompt, declared_variables = _load_prompt(root, name, config)
        prompts[name] = prompt
        variables[name] = declared_variables
        versions[name] = prompt.version
    return prompts, variables, versions


def _load_versioned_messages(
    root: Path,
    message_config: dict[str, Any],
) -> tuple[dict[str, dict[str, str]], str]:
    _validate_version_files(root, "messages", message_config)
    messages_version, messages_entry = _version_entry("messages", message_config)
    for version_metadata in message_config["versions"].values():
        _load_messages(root / version_metadata["file"])
    messages = _load_messages(root / messages_entry["file"])
    return messages, messages_version


def _load_catalog(root: Path) -> _Catalog:
    manifest = _read_toml(root / "manifest.toml")
    prompt_configs = manifest.get("prompts")
    message_config = manifest.get("messages")
    if not isinstance(prompt_configs, dict) or not isinstance(message_config, dict):
        raise ValueError("manifest must define prompts and messages")
    prompts, variables, versions = _load_prompts(root, prompt_configs)
    messages, messages_version = _load_versioned_messages(root, message_config)
    versions["messages"] = messages_version
    return _Catalog(prompts=prompts, variables=variables, messages=messages, versions=versions)


def _prompt_set_for(versions: dict[str, str]) -> str:
    serialized_versions = "\n".join(
        f"{name}@{version}" for name, version in sorted(versions.items())
    )
    return hashlib.sha256(serialized_versions.encode("utf-8")).hexdigest()[:8]


_CATALOG = _load_catalog(_PROMPT_DIR)


def get(name: str) -> Prompt:
    """Return the active version of a named prompt."""
    return _CATALOG.prompts[name]


def render(name: str, **variables: object) -> str:
    """Render a named prompt after checking its declared variables."""
    expected = _CATALOG.variables[name]
    provided = frozenset(variables)
    if provided != expected:
        raise ValueError(
            f"render variables for {name!r} are {sorted(provided)}, expected {sorted(expected)}"
        )
    return get(name).text.format(**variables)


def message(key: str, language: str) -> str:
    """Return a canned shopper- or model-facing line."""
    if language not in _LANGUAGES:
        raise KeyError(f"unknown language {language!r}")
    return _CATALOG.messages[key][language]


def active_versions() -> dict[str, str]:
    """Return active versions for prompts and canned messages."""
    return dict(_CATALOG.versions)


def prompt_set() -> str:
    """Return a stable short identifier for the active prompt versions."""
    return _prompt_set_for(_CATALOG.versions)
