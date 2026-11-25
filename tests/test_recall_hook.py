"""Hermes-local UserPromptSubmit recall hook: prose filter, query cleanup, rerank."""

import importlib.util
from pathlib import Path

_HOOK = Path(__file__).resolve().parents[1] / ".claude-plugin" / "hooks" / "mempal-recall-hook.py"
_spec = importlib.util.spec_from_file_location("mempal_recall_hook", _HOOK)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def test_prose_drops_tool_output_chrome_and_code():
    text = (
        "> 问题本身\n"
        '[PowerShell] {"command":"x"\n'
        "more cmd\n"
        "→ f   Length\n"
        ".claude\\a.md 794\n"
        "\n"
        "真正的答复在这里，这是一段足够长的中文说明。\n"
        "[Read D:/a.png]\n"
        "77: code\n"
        '{"found":true}\n'
        "[Since your last message: x]\n"
        "[Orca: Item 1 is In progress]\n"
        "结束语也保留。"
    )
    assert (
        hook._prose(text)
        == "> 问题本身\n真正的答复在这里，这是一段足够长的中文说明。\n结束语也保留。"
    )


def test_clean_query_strips_orca_envelope():
    prompt = "──── From: ORCA_SESSION_ID:X (q) ────\nSubject: 卡 1 问题\n正文 [Image #1]"
    assert hook._clean_query(prompt) == "卡 1 问题 正文"


def test_hits_skip_diary_echo_and_duplicates(monkeypatch):
    answer = (
        "结论：不改，继续用 automations run，理由和重开条件已落卡，原生事件给的三样我们都用不上。"
    )
    results = [
        {"room": "diary", "similarity": 0.99, "text": "CHECKPOINT:2026-11-23|session:x|recent:..."},
        {"room": "technical", "similarity": 0.95, "text": "> 钉钉叫醒要不要改回原生事件？"},
        {"room": "technical", "similarity": 0.80, "text": answer},
        {"room": "technical", "similarity": 0.79, "text": answer},
    ]
    monkeypatch.setattr(hook, "_search", lambda query, limit=3: results)
    hits = hook._hits("钉钉叫醒要不要改回原生事件？")
    assert [h["text"] for h in hits] == [answer]
