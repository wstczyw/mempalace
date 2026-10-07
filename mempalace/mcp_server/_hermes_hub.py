# Loaded into mempalace.mcp_server via exec (see __init__.py). Not a standalone module.
if __name__ != "mempalace.mcp_server":
    raise ImportError(f"{__name__} is an implementation fragment; import mempalace.mcp_server")

# Hermes-local fragment: the Hub daemon-job execution channel.
#
# Why this file exists separately from http.py
# --------------------------------------------
# `Hermes-MemPalace-Hub` and `Hermes-MemPalace-Warm-Host` both fire on logon
# with no ordering. `mempalace daemon start` calls
# daemon._discover_hub_job_proxy(); when that returns None the daemon falls back
# to seizing the palace writer lock, which the Hub already holds, and exits with
# "writable daemon startup refused" (upstream v3.10.0 daemon.py still raises
# exactly that). With the daemon dead and write_routing.hooks = require, every
# auto-save is blocked instead of saved.
#
# The two routes below are the fix, originally added by the local commit
# ddc0ea7 "feat(mcp): 统一共享 Hub 并补齐现代协议握手" directly inside the old
# single-file mempalace/mcp_server.py. Upstream v3.10.0 split that module into a
# package and deleted the file, which silently dropped the handlers while
# daemon.py kept calling them — every probe answered 404 and the daemon never
# started.
#
# Keeping the patch in its own fragment means a future upstream merge conflicts
# on one line of __init__.py's _FRAGMENTS tuple instead of on 60 lines inside an
# upstream-owned file. Nothing here modifies upstream source; it wraps the two
# entry points by rebinding package globals, which works because fragments exec
# into the same namespace and do_GET resolves _http_handle_get at call time.

import hmac as _hermes_hmac


def _http_internal_request_rejected(handler) -> bool:
    """Protect daemon-to-Hub execution with loopback plus a private token."""
    peer = handler.client_address[0] if handler.client_address else ""
    if not _http_is_loopback(peer):
        handler.send_error(403, "Forbidden")
        return True
    token = getattr(handler.server, "internal_token", "")
    provided = handler.headers.get("Authorization", "")
    if not token or not _hermes_hmac.compare_digest(provided, f"Bearer {token}"):
        handler.send_error(401, "Unauthorized")
        return True
    return False


def _http_handle_internal_daemon_job(handler) -> None:
    """Execute an exact queued job inside the Hub's single-writer boundary."""
    if handler._request_rejected(require_auth=False) or _http_internal_request_rejected(handler):
        return
    try:
        content_length = int(handler.headers.get("Content-Length", "0") or "0")
    except (TypeError, ValueError):
        content_length = 0
    if content_length < 0 or content_length > _HTTP_MAX_REQUEST_BYTES:
        handler._send_json(413, {"error": "Request too large"})
        return
    try:
        raw = handler.rfile.read(content_length)
        request = json.loads(raw.decode("utf-8"))
        kind = request.get("kind") if isinstance(request, dict) else None
        payload = request.get("payload") if isinstance(request, dict) else None
        if not isinstance(kind, str) or not kind or not isinstance(payload, dict):
            raise ValueError("kind must be non-empty and payload must be an object")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        handler._send_json(400, {"error": str(exc)})
        return
    from mempalace.service import execute_job

    if kind == "diary_write":
        # A daemon autosave uses the same explicit-vector embedding path as the
        # public MCP tool.  Give only that embedded interval to concurrent
        # readers while retaining the backend lifecycle lease until the write
        # has committed.  Mine and every other daemon job keep the pre-existing
        # whole-job writer boundary.
        # The 3.11 HTTP dispatcher installs this hook around the same
        # request lock.  Keep the Hub route on that public boundary so the
        # embedding section can yield the writer lease safely.
        with _http_embedding_release_installed("write"), _HTTP_REQUEST_LOCK:
            result = execute_job(kind, payload)
    else:
        with _HTTP_REQUEST_LOCK:
            result = execute_job(kind, payload)
    handler._send_json(200, {"result": result})


_hermes_upstream_handle_get = _http_handle_get


def _http_handle_get(handler) -> None:
    """GET /internal/daemon-job/health, then upstream's route chain."""
    if urlparse(handler.path).path == "/internal/daemon-job/health":
        if not handler._request_rejected(
            require_auth=False
        ) and not _http_internal_request_rejected(handler):
            handler._send_json(200, {"ok": True, "execution_owner": "hub"})
        return
    _hermes_upstream_handle_get(handler)


_hermes_upstream_build_http_server = _build_http_server


def _build_http_server(host: str, port: int):
    """Upstream's server plus the internal token and the daemon-job POST route.

    do_POST is a method on a class created inside upstream's _build_http_server,
    so it cannot be rebound at module level like _http_handle_get. Patch the
    class on the returned instance instead. The branch must run before
    upstream's `_request_rejected(require_auth=True)`, because the daemon
    authenticates with the private internal token, not the public bearer.
    """
    httpd = _hermes_upstream_build_http_server(host, port)

    # Reuse the palace's published credential(s) so the daemon reads the same
    # value from server_registry; do not mint here, because upstream keeps a
    # local-only loopback hub tokenless and must not write a token file.
    from mempalace.server_registry import load_server_tokens

    internal_tokens = load_server_tokens(_config.palace_path)
    httpd.internal_token = internal_tokens[0] if internal_tokens else getattr(
        httpd, "auth_token", ""
    )

    handler_cls = httpd.RequestHandlerClass
    if not getattr(handler_cls, "_hermes_daemon_job_patched", False):
        _upstream_do_POST = handler_cls.do_POST

        def do_POST(self):
            if urlparse(self.path).path == "/internal/daemon-job":
                _http_handle_internal_daemon_job(self)
                return
            _upstream_do_POST(self)

        handler_cls.do_POST = do_POST
        handler_cls._hermes_daemon_job_patched = True

    return httpd
