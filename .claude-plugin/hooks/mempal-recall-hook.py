"""UserPromptSubmit recall: history-style prompts get the top palace hits as context.

Hermes-local. The upstream hooks only write (Stop/PreCompact/SessionEnd); nothing
reads the palace back unless the model remembers to call mempalace_search, which
in 30 days happened 23 times against 72k drawers. This closes the read side.

Fail-open by design: any error, non-history prompt, or a Hub slower than the
budget (a mine holds the embedding model) prints nothing and exits 0.
"""

import json
import os
import re
import sys
import time
import urllib.request

HUB = os.environ.get("MEMPALACE_RECALL_URL", "http://127.0.0.1:8765/mcp")
BUDGET_SECONDS = float(os.environ.get("MEMPALACE_RECALL_TIMEOUT", "4"))
LIMIT = 3
SNIPPET_CHARS = 240
# Questions about the past: where recall changes the first action.
TRIGGER = re.compile(
    r"上次|之前|以前|当时|先例|历史|记得|为什么|为啥|根因|怎么做的|原来|裁定|决定过|老办法|又出现|又坏"
    r"|\blast time\b|\bpreviously\b|\bwhy\b|\bprecedent\b|\bremember\b",
    re.IGNORECASE,
)


def _search(query: str) -> list:
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "mempalace_search", "arguments": {"query": query, "limit": LIMIT}},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        HUB,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
    )
    with urllib.request.urlopen(req, timeout=BUDGET_SECONDS) as resp:
        raw = resp.read().decode("utf-8")
    if raw.lstrip().startswith("event:") or "\ndata:" in raw:
        raw = next(line[5:] for line in raw.splitlines() if line.startswith("data:"))
    payload = json.loads(json.loads(raw)["result"]["content"][0]["text"])
    return payload.get("results") or []


def _line(hit: dict) -> str:
    text = str(hit.get("text") or hit.get("content") or hit.get("document") or "")
    text = " ".join(text.split())[:SNIPPET_CHARS]
    where = "/".join(str(hit[k]) for k in ("wing", "room") if hit.get(k))
    src = os.path.basename(str(hit.get("source_file") or ""))
    return f"- [{where}{' ' + src if src else ''}] {text}"


def _log(outcome: str, started: float) -> None:
    # Side effect that proves the hook fired: one line per triggered prompt.
    try:
        path = os.path.join(os.path.expanduser("~"), ".mempalace", "hook_state", "recall.log")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {outcome} {time.monotonic() - started:.1f}s\n")
    except OSError:
        pass


def main() -> int:
    try:
        prompt = str(json.load(sys.stdin).get("prompt") or "")
    except Exception:
        return 0
    if len(prompt) < 4 or not TRIGGER.search(prompt):
        return 0
    started = time.monotonic()
    try:
        hits = _search(prompt[:400])
    except Exception as exc:
        _log(f"skip:{type(exc).__name__}", started)
        return 0
    lines = [_line(h) for h in hits if isinstance(h, dict)]
    lines = [line for line in lines if len(line) > 8]
    _log(f"hits:{len(lines)}", started)
    if not lines:
        return 0
    context = (
        "MemPalace 自动回忆（按提问检索的前几条，可能不相关；要引用先 mempalace_get_drawer 核原文）：\n"
        + "\n".join(lines)
    )
    sys.stdout.reconfigure(encoding="utf-8")
    print(
        json.dumps(
            {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
