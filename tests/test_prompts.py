import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent.trace import build_voice_trace
from ravo_core import assistant, monitor, prompts, understanding

SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "prompts_v1_snapshot.json"


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def test_v1_prompts_match_snapshot(snapshot: dict) -> None:
    assert prompts.get("shopper_rules").text == snapshot["shopper_rules"]
    assert prompts.get("relevance_check").text == snapshot["relevance_check"]
    assert (
        prompts.get("shopper_rules").text + prompts.get("voice_style").text
        == snapshot["voice_prompt"]
    )
    assert prompts.render(
        "relevance_check",
        query="laptop",
        names="- Cooler Bag\n- USB Cable",
    ) == snapshot["relevance_check_rendered_example"]


@pytest.mark.parametrize(
    ("key", "languages"),
    [
        ("weak_match_note", ("en",)),
        ("nothing_sold", ("en",)),
        ("language_instruction", ("ar", "en")),
        ("clarification", ("ar", "en")),
        ("greeting", ("en",)),
        ("goodbye", ("en",)),
        ("didnt_catch", ("ar", "en")),
    ],
)
def test_v1_messages_match_snapshot(
    snapshot: dict,
    key: str,
    languages: tuple[str, ...],
) -> None:
    expected = snapshot[key]
    for language in languages:
        value = expected if isinstance(expected, str) else expected[language]
        assert prompts.message(key, language) == value


def _write_catalog(
    tmp_path: Path,
    template: str,
    variables: str = 'variables = ["name"]',
) -> None:
    (tmp_path / "sample.v1.md").write_text(template, encoding="utf-8")
    (tmp_path / "messages.v1.toml").write_text(
        "[shopper]\n[model]\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.toml").write_text(
        "\n".join(
            [
                "[prompts.sample]",
                'active = "v1"',
                variables,
                "[prompts.sample.versions.v1]",
                'file = "sample.v1.md"',
                'note = "test"',
                "[messages]",
                'active = "v1"',
                "[messages.versions.v1]",
                'file = "messages.v1.toml"',
                'note = "test"',
            ]
        ),
        encoding="utf-8",
    )


def test_manifest_rejects_placeholder_mismatch(tmp_path: Path) -> None:
    _write_catalog(tmp_path, "Hello {unexpected}\n")

    with pytest.raises(ValueError, match="template variables"):
        prompts._load_catalog(tmp_path)


def test_manifest_rejects_unknown_language(tmp_path: Path) -> None:
    _write_catalog(tmp_path, "Hello {name}\n")
    (tmp_path / "messages.v1.toml").write_text(
        '[shopper.greeting]\nfr = "Bonjour"\n[model]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown language key"):
        prompts._load_catalog(tmp_path)


def test_manifest_rejects_missing_version_file(tmp_path: Path) -> None:
    _write_catalog(tmp_path, "Hello {name}\n")
    (tmp_path / "sample.v1.md").unlink()

    with pytest.raises(FileNotFoundError):
        prompts._load_catalog(tmp_path)


@pytest.mark.parametrize(
    ("stored", "loaded"),
    [("line", "line"), ("line\n", "line"), ("line\n\n", "line\n")],
)
def test_markdown_strips_exactly_one_trailing_newline(
    tmp_path: Path,
    stored: str,
    loaded: str,
) -> None:
    path = tmp_path / "prompt.md"
    path.write_text(stored, encoding="utf-8")

    assert prompts._read_markdown(path) == loaded


def test_prompt_set_is_stable_and_tracks_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    original_versions = prompts.active_versions()
    original_id = prompts.prompt_set()
    reordered = dict(reversed(list(original_versions.items())))
    monkeypatch.setattr(prompts, "_CATALOG", replace(prompts._CATALOG, versions=reordered))
    reordered_id = prompts.prompt_set()

    assert reordered_id == original_id

    changed = {**original_versions, "voice_style": "v2"}
    monkeypatch.setattr(prompts, "_CATALOG", replace(prompts._CATALOG, versions=changed))

    assert prompts.prompt_set() != reordered_id


def test_prompt_versions_are_added_to_typed_and_voice_traces() -> None:
    typed = assistant.run_turn([{"role": "user", "content": ""}])["trace"]
    spoken = build_voice_trace({}, {"e2e_latency": 0.5}, [])

    assert typed["prompts"] == prompts.active_versions()
    assert typed["prompt_set"] == prompts.prompt_set()
    assert spoken is not None
    assert spoken["prompts"] == prompts.active_versions()
    assert spoken["prompt_set"] == prompts.prompt_set()


def test_monitor_summary_keeps_prompt_set() -> None:
    trace = {
        "steps": [],
        "total_ms": 1.0,
        "prompts": prompts.active_versions(),
        "prompt_set": prompts.prompt_set(),
    }

    summary = monitor.summarize(trace)

    assert summary["prompt_set"] == prompts.prompt_set()
    assert "prompts" not in summary


def test_public_prompt_bindings_match_snapshot(snapshot: dict) -> None:
    assert assistant.BASE_RULES == snapshot["shopper_rules"]
    assert assistant.RELEVANCE_PROMPT == snapshot["relevance_check"]
    assert assistant.WEAK_MATCH_NOTE == snapshot["weak_match_note"]
    assert assistant.NOTHING_SOLD == snapshot["nothing_sold"]
    assert assistant.language_instruction("عايز لبن") == snapshot["language_instruction"]["ar"]
    assert assistant.language_instruction("milk") == snapshot["language_instruction"]["en"]
    assert understanding.clarification("ar") == snapshot["clarification"]["ar"]
    assert understanding.clarification("en") == snapshot["clarification"]["en"]
