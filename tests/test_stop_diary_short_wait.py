"""Stop diary writes wait a short budget, then the payload is on disk.

A queued daemon used to hold Stop for the full 30s diary wait. Stop now waits
2s (``MEMPALACE_HOOK_DIARY_WAIT_SECS``). Timeout or failure fsyncs the full
entry to ``pending_diary.jsonl`` before the hook returns. The next save
flushes that file inside the same short budget. SessionEnd and PreCompact
keep the original 30s wait.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from mempalace import hooks_cli
from mempalace.hooks_cli import (
    SAVE_INTERVAL,
    _flush_pending_diary,
    _pending_diary_record,
    _stop_diary_wait_secs,
    hook_session_end,
    hook_stop,
)

_TOKEN = "STOPFIX-PAYLOAD"
_PARKED = {
    "id": "job-parked",
    "state": "queued",
    "error": {
        "error_class": "LockHeldByOtherProcess",
        "message": "palace /p is held by PID 999 (mempalace-mcp)",
    },
    "result": None,
}


class _HookConfig:
    hooks_auto_save = True
    hook_silent_save = True
    hook_desktop_toast = False
    hook_use_daemon = True
    palace_path = "/tmp/palace"


@pytest.fixture
def hook_state(tmp_path, monkeypatch):
    root = tmp_path / ".mempalace"
    state = root / "hook_state"
    state.mkdir(parents=True)
    monkeypatch.setattr(hooks_cli, "PALACE_ROOT", root)
    monkeypatch.setattr(hooks_cli, "STATE_DIR", state)
    monkeypatch.setattr(hooks_cli, "_MINE_PID_DIR", state / "mine_pids")
    monkeypatch.setattr(hooks_cli, "_state_dir_initialized", False)
    for key in (
        "MEMPALACE_HOOK_DIARY_WAIT_SECS",
        "MEMPALACE_WRITE_ROUTING",
        "MEMPALACE_HOOK_WRITE_ROUTING",
        "MEMPALACE_HOOKS_DAEMON",
    ):
        monkeypatch.delenv(key, raising=False)
    return state


def _write_transcript(path: Path, count: int, last: str) -> None:
    rows = [{"message": {"role": "user", "content": f"note {index}"}} for index in range(count - 1)]
    rows.append({"message": {"role": "user", "content": last}})
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _queue(state: Path, session_id: str, entry: str) -> dict:
    record = _pending_diary_record(
        session_id,
        {
            "agent_name": "claude",
            "entry": entry,
            "topic": "checkpoint",
            "wing": "wing_sessions",
        },
    )
    path = state / "pending_diary.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def _succeeded(entry_id: str = "e1") -> dict:
    return {
        "id": "job",
        "state": "succeeded",
        "result": {"success": True, "entry_id": entry_id},
    }


@contextmanager
def _routed(submit):
    with (
        patch("mempalace.hooks_cli.MempalaceConfig", return_value=_HookConfig()),
        patch("mempalace.hooks_cli._daemon_available", return_value=True),
        patch("mempalace.hooks_cli._submit_daemon_job", side_effect=submit),
        patch("mempalace.hooks_cli._ingest_transcript"),
        patch("mempalace.hooks_cli._maybe_auto_ingest"),
        patch("mempalace.hooks_cli._output"),
        patch("mempalace.hooks_cli.subprocess.Popen"),
    ):
        yield


def _stop(transcript: Path, session_id: str) -> None:
    hook_stop(
        {
            "session_id": session_id,
            "transcript_path": str(transcript),
            "stop_hook_active": False,
        },
        "claude-code",
    )


def test_stop_timeout_persists_full_diary_and_returns_within_3s(hook_state, tmp_path):
    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, SAVE_INTERVAL, _TOKEN)
    pending = hook_state / "pending_diary.jsonl"
    seen: list[float] = []
    existed: list[bool] = []

    def submit(kind, payload, **kwargs):
        timeout = float(kwargs["timeout"])
        seen.append(timeout)
        existed.append(pending.exists())
        if timeout > 2.5:
            raise TimeoutError("timed out waiting for job")
        time.sleep(timeout)
        raise TimeoutError("timed out waiting for job")

    started = time.monotonic()
    with _routed(submit):
        _stop(transcript, "stop-tail")
    elapsed = time.monotonic() - started

    assert elapsed < 3.0
    assert len(seen) == 1
    assert 1.5 <= seen[0] <= 2.0
    assert existed == [False]
    assert not (hook_state / "last_checkpoint").exists()
    lines = [line for line in pending.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    entry = row["payload"]["entry"]
    assert row["session_id"] == "stop-tail"
    assert row["payload"]["agent_name"] == "claude"
    assert row["payload"]["topic"] == "checkpoint"
    assert _TOKEN in entry
    assert row["idempotency_key"] == (
        "stop-tail:" + hashlib.sha256(entry.encode("utf-8")).hexdigest()
    )
    print("STOPFIX OK")


def test_next_stop_flushes_pending_diary(hook_state, tmp_path):
    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, SAVE_INTERVAL, "fresh-checkpoint")
    queued = _queue(hook_state, "earlier", "entry-from-previous-stop")
    calls: list[str] = []

    def submit(kind, payload, **kwargs):
        assert kind == "diary_write"
        calls.append(kwargs["dedupe_key"])
        assert payload["entry"]
        return _succeeded()

    with _routed(submit):
        _stop(transcript, "stop-next")
        _flush_pending_diary(2)

    assert calls[0] == queued["idempotency_key"]
    assert calls.count(queued["idempotency_key"]) == 1
    assert len(calls) == 2
    assert (hook_state / "pending_diary.jsonl").read_text(encoding="utf-8").strip() == ""


def test_repeat_flush_writes_each_idempotency_key_once(hook_state):
    first = _queue(hook_state, "sess", "same-entry")
    _queue(hook_state, "sess", "same-entry")
    failed = _queue(hook_state, "sess", "needs-a-retry")
    calls: list[str] = []
    fail = True

    def submit(kind, payload, **kwargs):
        key = kwargs["dedupe_key"]
        calls.append(key)
        if key == failed["idempotency_key"] and fail:
            raise TimeoutError("timed out waiting for job")
        return _succeeded()

    with _routed(submit):
        confirmed = _flush_pending_diary(2)
        text = (hook_state / "pending_diary.jsonl").read_text(encoding="utf-8")
        fail = False
        again = _flush_pending_diary(2)
        third = _flush_pending_diary(2)

    assert confirmed == {first["idempotency_key"]}
    assert calls.count(first["idempotency_key"]) == 1
    assert json.loads(text)["idempotency_key"] == failed["idempotency_key"]
    assert again == {failed["idempotency_key"]}
    assert calls.count(failed["idempotency_key"]) == 2
    assert third == set()
    assert calls.count(first["idempotency_key"]) == 1
    assert (hook_state / "pending_diary.jsonl").read_text(encoding="utf-8").strip() == ""


def test_flush_shares_one_short_budget(hook_state):
    for index in range(3):
        _queue(hook_state, "sess", f"queued-{index}")
    waits: list[float] = []

    def submit(kind, payload, **kwargs):
        timeout = float(kwargs["timeout"])
        waits.append(timeout)
        if timeout > 0.5:
            raise TimeoutError("timed out waiting for job")
        time.sleep(timeout)
        raise TimeoutError("timed out waiting for job")

    started = time.monotonic()
    with _routed(submit):
        _flush_pending_diary(0.25)
    elapsed = time.monotonic() - started

    assert waits
    assert sum(waits) < 0.45
    assert elapsed < 1.0
    lines = (hook_state / "pending_diary.jsonl").read_text(encoding="utf-8").splitlines()
    assert len([line for line in lines if line.strip()]) == 3


def test_session_end_and_precompact_keep_30s_diary_wait(hook_state, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMPALACE_HOOK_DIARY_WAIT_SECS", "0.25")
    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, 2, "session-end-note")
    timeouts: list[float] = []

    def submit(kind, payload, **kwargs):
        if kind == "diary_write":
            timeouts.append(float(kwargs["timeout"]))
        return _succeeded()

    with _routed(submit):
        hook_session_end(
            {
                "session_id": "sess-end",
                "transcript_path": str(transcript),
                "stop_hook_active": False,
            },
            "claude-code",
        )
        hooks_cli._save_diary_direct(
            str(transcript),
            "sess-pre",
            agent_name="claude",
            hook_event="precompact",
        )

    assert timeouts == [30, 30]


def test_stop_diary_wait_env_override(monkeypatch):
    monkeypatch.delenv("MEMPALACE_HOOK_DIARY_WAIT_SECS", raising=False)
    assert _stop_diary_wait_secs() == 2.0
    monkeypatch.setenv("MEMPALACE_HOOK_DIARY_WAIT_SECS", "0.4")
    assert _stop_diary_wait_secs() == 0.4
    monkeypatch.setenv("MEMPALACE_HOOK_DIARY_WAIT_SECS", "0")
    assert _stop_diary_wait_secs() == 2.0
    monkeypatch.setenv("MEMPALACE_HOOK_DIARY_WAIT_SECS", "nope")
    assert _stop_diary_wait_secs() == 2.0


def test_lock_deferral_is_not_copied_into_pending(hook_state, tmp_path):
    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, SAVE_INTERVAL, "parked-entry")

    def submit(kind, payload, **kwargs):
        return _PARKED

    with _routed(submit):
        _stop(transcript, "stop-parked")

    assert not (hook_state / "pending_diary.jsonl").exists()
    assert not (hook_state / "last_checkpoint").exists()
