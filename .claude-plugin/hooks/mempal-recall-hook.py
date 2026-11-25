"""UserPromptSubmit recall: history-style prompts get the top palace hits as context.

Hermes-local. The upstream hooks only write (Stop/PreCompact/SessionEnd); nothing
reads the palace back unless the model remembers to call mempalace_search, which
in 30 days happened 23 times against 72k drawers. This closes the read side.

Recall hygiene (11-25): the palace holds transcript chunks with tool calls, Orca
chrome and diary checkpoints. Search stays on the verbatim drawers; the hook only
shows the prose lines of a hit and skips hits that are diary checkpoints, system
chrome, tool output, or the prompt itself echoed back.

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
CANDIDATES = int(os.environ.get("MEMPALACE_RECALL_CANDIDATES", "30"))
OVERLAP_WEIGHT = (
    0.2  # rerank: similarity + weight * share of the question's terms found in the prose
)
QUESTION_PENALTY = 0.05  # another ask (quoted prompt, request) ranks below a statement or answer
MIN_PROSE = 40  # chars of real prose a hit needs once tool output and chrome are gone
SNIPPET_CHARS = 240
QUERY_CHARS = 250  # mempalace_search rejects longer queries ("must be at most 250 characters")
# Wings kept out of auto-recall; mempalace_search with wing=<name> still finds them.
# people: colleague cards from the Obsidian 70-People folder, searched on purpose only.
EXCLUDED_WINGS = frozenset(
    w.strip()
    for w in os.environ.get("MEMPALACE_RECALL_EXCLUDE_WINGS", "people").split(",")
    if w.strip()
)
# Questions about the past: where recall changes the first action.
TRIGGER = re.compile(
    r"上次|之前|以前|当时|先例|历史|记得|为什么|为啥|根因|怎么做的|原来|裁定|决定过|老办法|又出现|又坏"
    r"|\blast time\b|\bpreviously\b|\bwhy\b|\bprecedent\b|\bremember\b",
    re.IGNORECASE,
)
# Orca message envelope and injected status lines (not anyone's words).
_ENVELOPE = re.compile(
    r"^(?:> )?(?:──── From: |\[Reply: |\[To answer its sender|\[Orca[: ]|Orca restarted\. "
    r"|Subject: Your (?:other chat|owner) |Your (?:other chat|owner) (?:changed|updated|marked|commented))"
)
_CHROME = re.compile(
    r"\[Orca[: ]|Orca: if you asked|is In progress but nothing is running|Next: .{0,80}wakes you"
    r"|Stop hook feedback|\[Since your last message|\[To answer its sender|\[Reply: orca "
    r"|<system-reminder>|<task-notification>|do not report, assume, or predict"
    r"|Do not duplicate this agent's work|^output_file: |taskId=\S+ runId=\S+"
    r"|Async agent launched|Do NOT Read or tail this file|If the user asks for progress|working in the background"
    r"|CHECKPOINT:\d{4}-\d\d-\d\d\|session:"
)
# Tool-call lines: bracketed tool name, then the command, JSON args, or a path.
_TOOL_CALL = re.compile(r"^(?:> )?\[(?:[A-Z][\w.:-]*|mcp__[\w-]+)(?:\] | [^\]]*\]$)")
# Lines that are code, paths, listings or JSON rather than anyone's prose.
_CODE = re.compile(
    r"^\d+[:|]|^[{\[]\"|\\u[0-9a-f]{4}|2>&1|\| *(?:cut|head|tail|grep|sed|jq)\b|^[A-Za-z]:[\\/]\S*$"
    r"|^[\w./\\-]+\.(?:py|md|ps1|ts|js|json|jsonl|png|txt|csv)$|^[0-9a-f]{8}-[0-9a-f]{4}-|^-+ +-+"
    r"|^[A-Za-z]:[\\/][^\s:]*:\d+:| && |^\"bot:|address: bot:|\"\w+\":\""
)
_ITEM_WRAPPER = re.compile(r'(?m)^- item \d+ "[^"\n]*": commented \(event \w+\): <<<\s*')
_SENTENCE_END = re.compile(r"(?<=[。？！?!；;])")


def _search(query: str, limit: int = LIMIT) -> list:
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "mempalace_search", "arguments": {"query": query, "limit": limit}},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        HUB,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=BUDGET_SECONDS) as resp:
        raw = resp.read().decode("utf-8")
    if raw.lstrip().startswith("event:") or "\ndata:" in raw:
        raw = next(line[5:] for line in raw.splitlines() if line.startswith("data:"))
    reply = json.loads(raw)
    if "error" in reply:
        raise RuntimeError(str(reply["error"].get("message") or reply["error"])[:200])
    payload = json.loads(reply["result"]["content"][0]["text"])
    return payload.get("results") or []


def _clean_query(prompt: str) -> str:
    """The question without the Orca envelope, so the 250-char query is the ask."""
    lines = []
    for line in prompt.splitlines():
        if line.startswith("Subject: "):
            line = line[len("Subject: ") :]
        elif _ENVELOPE.match(line):
            continue
        line = re.sub(r"\[Image #\d+\]", "", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines)[:QUERY_CHARS] or prompt[:QUERY_CHARS]


def _prose(text: str) -> str:
    """Verbatim prose lines of a drawer: tool calls (and their continuation
    lines), tool output lines (arrow-prefixed), envelopes and chrome dropped."""
    keep, mode = [], ""  # "call" runs to its arrow output; "out" runs to a blank line
    for line in text.splitlines():
        s = line.strip()
        if _TOOL_CALL.match(s):
            mode = "" if s.endswith("]") else "call"  # "[Read D:/a.png]" has no output block
        elif s.startswith("→"):
            mode = "out"
        elif not s:
            mode = "" if mode == "out" else mode
        elif mode or _ENVELOPE.match(s) or _CHROME.search(s) or _CODE.search(s) or _symbolic(s):
            continue
        else:
            keep.append(s)
    return "\n".join(keep)


def _symbolic(s: str) -> bool:
    """JSON, HTML or code: more than 1 in 8 characters is markup punctuation."""
    return len(s) >= 20 and sum(c in '{}[]"<>=\\/_;$%&' for c in s) * 8 > len(s)


def _terms(s: str) -> set:
    """Chinese character pairs plus ASCII words of 3+ letters, for a keyword-overlap rerank."""
    s = s.lower()
    cjk = re.findall(r"[一-鿿]+", s)
    return {run[i : i + 2] for run in cjk for i in range(len(run) - 1)} | set(
        re.findall(r"[a-z][a-z0-9_.]{2,}|\d{3,}", s)
    )


def _norm(s: str) -> str:
    return re.sub(r"[\s>*`|#\-]+", "", s)


def _hits(prompt: str) -> list:
    query = _clean_query(prompt)
    asked = _norm(prompt)
    want = _terms(query)
    ranked = []
    for hit in _search(query, CANDIDATES):
        if not isinstance(hit, dict) or str(hit.get("room") or "") == "diary":
            continue
        if str(hit.get("wing") or "") in EXCLUDED_WINGS:
            continue
        raw = str(hit.get("text") or hit.get("content") or hit.get("document") or "")
        # Drop sentences that only repeat the question being asked.
        parts = _SENTENCE_END.split(_prose(raw).replace("\n", "。\n"))
        rest = "".join(p for p in parts if len(_norm(p)) > 1 and _norm(p).rstrip("。") not in asked)
        rest = _ITEM_WRAPPER.sub("", rest.replace("。\n", "\n"))
        if len(_norm(rest)) < MIN_PROSE:
            continue
        overlap = len(want & _terms(rest)) / len(want) if want else 0.0
        head = rest[:200]
        asks = head.startswith(">") or (
            head.count("？") + head.count("?") > 0 and "Re:" not in head[:20]
        )
        score = (
            float(hit.get("similarity") or 0) + OVERLAP_WEIGHT * overlap - QUESTION_PENALTY * asks
        )
        ranked.append((score, dict(hit, text=rest)))
    out = []
    for _, hit in sorted(ranked, key=lambda r: -r[0]):
        key = _norm(hit["text"])[:60]
        if any(key in _norm(h["text"]) for h in out):
            continue  # the same passage stored twice
        out.append(hit)
        if len(out) == LIMIT:
            break
    return out


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
            fh.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} {outcome} {time.monotonic() - started:.1f}s\n"
            )
    except OSError:
        pass


def main() -> int:
    if os.environ.get("MEMPALACE_RECALL_DISABLE") == "1":
        return 0  # per-project opt-out: set in that project's .claude/settings.json env
    try:
        prompt = str(json.load(sys.stdin).get("prompt") or "")
    except Exception:
        return 0
    if len(prompt) < 4 or not TRIGGER.search(prompt):
        return 0
    started = time.monotonic()
    try:
        hits = _hits(prompt)
    except Exception as exc:
        _log(f"skip:{type(exc).__name__}:{str(exc)[:200]}", started)
        return 0
    lines = [_line(h) for h in hits]
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
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                }
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
