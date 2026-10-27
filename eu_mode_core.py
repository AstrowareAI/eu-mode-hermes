# Copyright 2026 Astroware
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EU Mode core: send the conversation to HaloGuard EU and apply its answer.

Standard library only, and nothing here imports Hermes, so it can be tested on
its own. The Hermes glue is ``__init__.py``. One user turn, as Hermes runs it:

    pre_llm_call          -> start_turn()      check the conversation ending in the
                                               user's message: decision + masked copy
    llm_request (each)    -> rewrite_request() put the masked copy into what is sent
                                               to the model; a blocked message is
                                               swapped for a fixed note
    transform_llm_output  -> finish_turn()     put the originals back into the reply,
                                               check the reply going out, add EU
                                               Mode's lines

What HaloGuard EU answers (``choices[0].message.content`` is JSON):
``decision`` ("allow" | "block") and ``policy`` always; ``label`` on some blocks.
When the last message is a user message: ``privacy_mode`` and normally
``masked_messages`` (the whole conversation, one shared set of placeholders),
``map`` (placeholder -> original) and ``personal_data``; or
``"privacy": "unavailable"`` when the filter could not answer. A model reply
going out is never masked.

Fails open: when HaloGuard EU cannot answer, the turn goes on unchecked and the
reply says so. Masking is never skipped silently: every turn where it did not
happen carries a line saying so. Nothing here logs or stores message text, and
the key is never logged or shown.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

log = logging.getLogger("eu-mode")

VERSION = "0.1.1"
USER_AGENT = "eu-mode-hermes/0.1"        # fixed, so the gateway can count EU Mode's calls
MODEL = "haloguard-eu"
DEFAULT_BASE_URL = "https://api.astroware.ai/v1"
KEY_ENV = "HALOGUARD_API_KEY"
DEFAULT_TIMEOUT = 5.0
TIMEOUT_RANGE = (0.5, 30.0)

PREFIX = "[EU Mode beta]"
# What the model gets in place of a message it must not see. Fixed text: nothing
# from the message, the policy or the API rides along.
REMOVED_NOTE = "[EU Mode removed this message: HaloGuard EU blocked it. Reply only: OK.]"
HELD_NOTE = "[EU Mode held this message back: personal data could not be masked. Reply only: OK.]"

TEXT_PART_TYPES = (None, "text", "input_text", "output_text")
PLACEHOLDER = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]")
ROLES = {"user": "user", "assistant": "assistant", "model": "assistant"}   # "model": Gemini's name for the assistant
MAX_SESSIONS = 64
MAX_MAPS = 2000


# ---------------------------------------------------------------- settings


def _on(value: Optional[str], default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    key: str = field(repr=False)          # never in a repr, a log line or an error
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    require_masking: bool = False
    footer: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        """Read the settings from ``env``: under Hermes, its secret scope (see ``use_env``)."""
        try:
            timeout = float(env.get("EU_MODE_TIMEOUT", "") or DEFAULT_TIMEOUT)
        except ValueError:
            timeout = DEFAULT_TIMEOUT
        timeout = min(max(timeout, TIMEOUT_RANGE[0]), TIMEOUT_RANGE[1])
        return cls(key=(env.get(KEY_ENV) or "").strip(),
                   base_url=(env.get("HALOGUARD_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/"),
                   timeout=timeout,
                   require_masking=_on(env.get("EU_MODE_REQUIRE_MASKING"), False),
                   footer=_on(env.get("EU_MODE_FOOTER"), True))

    def url_problem(self) -> Optional[str]:
        """The key only ever travels over HTTPS (plain HTTP only to this machine, for tests)."""
        u = urllib.parse.urlsplit(self.base_url)
        if u.scheme == "https" and u.hostname:
            return None
        if u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1", "::1"):
            return None
        return "HALOGUARD_BASE_URL must be an https:// address"


# ---------------------------------------------------------------- the API


@dataclass
class Answer:
    checked: bool                          # False: HaloGuard EU did not answer; the turn goes on (fails open)
    decision: str = "allow"
    policy: Optional[str] = None
    label: Optional[str] = None
    masked: Optional[list] = None          # masked text per message sent, same order; None = no masked copy
    map: dict = field(default_factory=dict)
    personal_data: list = field(default_factory=list)
    privacy_mode: Optional[str] = None
    privacy_kept: list = field(default_factory=list)
    privacy_unavailable: bool = False
    problem: Optional[str] = None          # why it was not checked, in words for the user


def _http_problem(err: urllib.error.HTTPError) -> str:
    detail = ""
    if err.code in (400, 429):             # our own API's message: a budget or size limit is worth showing
        try:
            msg = json.loads(err.read() or b"{}").get("error", {}).get("message", "")
            if isinstance(msg, str) and msg.strip():
                detail = ": " + " ".join(msg.split())[:160]
        except (ValueError, AttributeError, OSError):
            pass
    if err.code == 401:
        return "HaloGuard EU did not accept the key (HTTP 401)"
    return f"HaloGuard EU refused the check (HTTP {err.code}{detail})"


def ask(turns: list, settings: Settings, opener: Optional[Callable] = None) -> Answer:
    """One check. ``turns`` is [(role, text)], ending in the message to judge."""
    bad_url = settings.url_problem()
    if bad_url:
        return Answer(checked=False, problem=bad_url)
    body = json.dumps({"model": MODEL, "messages": [{"role": r, "content": t} for r, t in turns]}).encode()
    request = urllib.request.Request(
        settings.base_url + "/chat/completions", data=body, method="POST",
        headers={"Authorization": "Bearer " + settings.key, "Content-Type": "application/json",
                 "Accept": "application/json", "User-Agent": USER_AGENT})
    try:
        with (opener or urllib.request.urlopen)(request, timeout=settings.timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        try:
            return Answer(checked=False, problem=_http_problem(e))
        finally:
            e.close()
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            return Answer(checked=False, problem="HaloGuard EU did not answer in time")
        return Answer(checked=False, problem="HaloGuard EU could not be reached")
    except TimeoutError:
        return Answer(checked=False, problem="HaloGuard EU did not answer in time")
    except OSError:
        return Answer(checked=False, problem="HaloGuard EU could not be reached")
    try:
        a = json.loads(json.loads(raw)["choices"][0]["message"]["content"])
        if a["decision"] not in ("allow", "block"):
            raise ValueError("decision")
    except (ValueError, KeyError, IndexError, TypeError):
        return Answer(checked=False, problem="HaloGuard EU sent an answer EU Mode could not read")
    out = Answer(checked=True, decision=a["decision"],
                 policy=a.get("policy") if isinstance(a.get("policy"), str) else None,
                 label=a.get("label") if isinstance(a.get("label"), str) else None,
                 privacy_mode=a.get("privacy_mode") if isinstance(a.get("privacy_mode"), str) else None,
                 privacy_kept=[k for k in a.get("privacy_kept") or [] if isinstance(k, str)],
                 privacy_unavailable=a.get("privacy") == "unavailable")
    masked, mapping = a.get("masked_messages"), a.get("map")
    if masked is not None and not out.privacy_unavailable:
        ok = (isinstance(masked, list) and len(masked) == len(turns) and isinstance(mapping, dict)
              and all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in masked))
        if ok:
            out.masked = [m["content"] for m in masked]
            # Only placeholder-shaped keys: a map can never rewrite ordinary words in a reply.
            out.map = {k: v for k, v in mapping.items() if isinstance(k, str) and isinstance(v, str) and PLACEHOLDER.fullmatch(k)}
            out.personal_data = [p for p in a.get("personal_data") or [] if isinstance(p, str)]
        else:
            out.privacy_unavailable = True       # a masked copy EU Mode cannot use is no masked copy
    return out


# ---------------------------------------------------------------- conversation text


def text_of(content: Any) -> str:
    """The text of a message: a string, or the text parts of a list (images and tool blocks are left out)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p["text"] for p in content
                         if isinstance(p, dict) and p.get("type") in TEXT_PART_TYPES and isinstance(p.get("text"), str))
    return ""


def text_turns(messages: Any) -> list:
    """[(role, text)] for the user and assistant messages that have text. System prompts and tool
    results are left out: HaloGuard EU takes only user and assistant text, and they are not masked."""
    out = []
    for m in messages if isinstance(messages, list) else []:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            t = text_of(m.get("content"))
            if t.strip():
                out.append((m["role"], t))
    return out


def restore(text: str, mapping: Mapping[str, str]) -> str:
    """Put the originals back for the placeholders in ``text``."""
    for k in sorted(mapping, key=len, reverse=True):
        text = text.replace(k, mapping[k])
    return text


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


# ---------------------------------------------------------------- per-session state


@dataclass
class Turn:
    user_raw: str                           # the user's message as Hermes holds it
    sent: list                              # [(role, text)] as sent to HaloGuard EU (earlier replies with originals back)
    check: Optional[Answer] = None
    swaps: list = field(default_factory=list)   # [(role, text as held, text to send)]
    expect: Optional[str] = None            # what must replace the user's message in the request (masked copy or a note)
    applied: bool = False
    blocked: bool = False
    held: bool = False
    lines: list = field(default_factory=list)


@dataclass
class Session:
    turn: Optional[Turn] = None
    blocked: set = field(default_factory=set)        # sha of each blocked message (user's or model's), kept on disk
    removed: dict = field(default_factory=dict)      # text as held -> note, for every later request of the session
    maps: "OrderedDict[str, dict]" = field(default_factory=OrderedDict)  # sha of a reply as held -> its turn's map
    last_map: dict = field(default_factory=dict)
    said_off: bool = False
    said_side: bool = False


_lock = threading.Lock()
_store_lock = threading.Lock()
# Where settings come from. The Hermes glue sets this to Hermes's secret scope, which honours the
# active profile; EU Mode never reads the process environment itself. Empty: no key, so nothing is checked.
_env: Mapping[str, str] = {}


def use_env(source: Mapping[str, str]) -> None:
    global _env
    _env = source


def current_settings() -> "Settings":
    return Settings.from_env(_env)
_sessions: "OrderedDict[str, Session]" = OrderedDict()
status: dict = {}                            # the last check, for /eu-mode: no text, no key

# Where the fingerprints (SHA-256, never text) of blocked messages are kept, so a blocked message
# stays out of every later request even after Hermes restarts. None: kept in memory only.
_store_path: Optional[str] = None
MAX_STORED_SESSIONS = 500
MAX_STORED_PER_SESSION = 500


def use_store(path: Any) -> None:
    global _store_path
    _store_path = str(path) if path else None


def _read_store() -> dict:
    if not _store_path:
        return {}
    try:
        with open(_store_path, encoding="utf-8") as f:
            data = json.load(f)
        sessions = data.get("sessions") if isinstance(data, dict) else None
        return sessions if isinstance(sessions, dict) else {}
    except (OSError, ValueError):
        return {}


def _remember_blocked(session_id: str, sess: Session, text: str) -> None:
    h = _sha(text)
    sess.blocked.add(h)
    sess.removed[text] = REMOVED_NOTE
    if not _store_path:
        return
    with _store_lock:
        try:
            sessions = _read_store()
            kept = [x for x in sessions.pop(session_id, []) if isinstance(x, str) and x != h] + [h]
            sessions[session_id] = kept[-MAX_STORED_PER_SESSION:]
            while len(sessions) > MAX_STORED_SESSIONS:
                sessions.pop(next(iter(sessions)))
            os.makedirs(os.path.dirname(_store_path) or ".", mode=0o700, exist_ok=True)
            tmp = _store_path + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "sessions": sessions}, f)
            os.replace(tmp, _store_path)
        except OSError as e:
            log.warning("eu-mode: could not save a blocked message's fingerprint (%s); it is kept "
                        "out of requests until Hermes restarts", type(e).__name__)


def _session(session_id: str) -> Session:
    with _lock:
        s = _sessions.get(session_id)
        if s is None:
            s = _sessions[session_id] = Session()
            stored = _read_store().get(session_id)
            s.blocked = {h for h in stored if isinstance(h, str)} if isinstance(stored, list) else set()
            while len(_sessions) > MAX_SESSIONS:
                _sessions.popitem(last=False)
        else:
            _sessions.move_to_end(session_id)
        return s


def forget(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


def _line(text: str) -> str:
    return f"{PREFIX} {text}"


def _blocked_line(a: Answer, what: str) -> str:
    why = f" Policy: {a.policy}" + (f" ({a.label})" if a.label else "") + "." if a.policy else ""
    return _line(f"HaloGuard EU blocked {what}.{why}")


def _cannot_mask(turn: Turn, s: Settings, reason: str) -> None:
    """Masking was expected and did not happen: hold the message, or send it as written and say so."""
    if s.require_masking:
        turn.held, turn.expect = True, HELD_NOTE
        turn.lines.append(_line(f"{reason}, so your message was NOT sent to the model "
                                "(EU_MODE_REQUIRE_MASKING is on)."))
    else:
        turn.lines.append(_line(f"{reason}. Your message went to the model as written, not masked."))


def start_turn(session_id: str, user_message: str, history: Any, settings: Optional[Settings] = None,
               opener: Optional[Callable] = None, side_lines: Optional[list] = None) -> None:
    """Check the conversation that ends in the user's message. Called once per user turn.
    ``side_lines``: the host's own side requests that send text to the model around EU Mode
    (said once per session, in its first reply)."""
    s = settings or current_settings()
    sess = _session(session_id)
    # Replies written last turn were written against last turn's placeholders: remember which, so
    # their originals can go back before this turn's check (one shared set across the conversation).
    # A reply already in the history when EU Mode first saw it (a resumed session) gets no map.
    held = text_turns(history)
    for role, raw in held:
        h = _sha(raw)
        if h in sess.blocked:
            sess.removed[raw] = REMOVED_NOTE          # blocked earlier, maybe before a restart
        if role == "assistant" and h not in sess.maps:
            sess.maps[h] = sess.last_map if PLACEHOLDER.search(raw) else {}
    while len(sess.maps) > MAX_MAPS:
        sess.maps.popitem(last=False)
    sess.last_map = {}
    held = [(r, t) for r, t in held if t not in sess.removed]
    if not held or held[-1][0] != "user":
        if not isinstance(user_message, str) or not user_message.strip():
            sess.turn = None                  # no text to check (an image only, say): EU Mode checks and masks text
            return
        held.append(("user", user_message))
    sent = [(r, restore(t, sess.maps.get(_sha(t), {})) if r == "assistant" else t) for r, t in held]
    turn = sess.turn = Turn(user_raw=held[-1][1], sent=sent)
    if side_lines and not sess.said_side:
        sess.said_side = True
        turn.lines.extend(_line(t) for t in side_lines)

    if not s.key:
        status.update(decision=None, problem=f"{KEY_ENV} is not set", privacy_mode=None, masked=[])
        if s.require_masking:
            _cannot_mask(turn, s, f"{KEY_ENV} is not set")
        else:
            turn.lines.append(_line(f"{KEY_ENV} is not set. Nothing was checked or masked."))
        return

    a = turn.check = ask(sent, s, opener)
    status.update(decision=a.decision if a.checked else None, problem=a.problem,
                  privacy_mode=a.privacy_mode, masked=list(a.personal_data))
    if not a.checked:
        log.warning("eu-mode: message not checked (%s)", a.problem)
        if s.require_masking:
            _cannot_mask(turn, s, f"{a.problem}. Nothing was checked")
        else:
            turn.lines.append(_line(f"{a.problem}. Your message went to the model unchecked and not masked."))
        return
    if a.decision == "block":
        turn.blocked, turn.expect = True, REMOVED_NOTE
        _remember_blocked(session_id, sess, turn.user_raw)
        log.info("eu-mode: user message blocked (policy %s)", a.policy)
        return
    if a.masked is not None:
        sess.last_map = dict(a.map)
        turn.swaps = [(r, raw, m) for (r, raw), m in zip(held, a.masked) if raw != m]
        mine = a.masked[-1]
        if mine != turn.user_raw:
            turn.expect = mine
        if s.footer and a.personal_data:
            kinds = ", ".join(dict.fromkeys(a.personal_data))
            kept = f" Kept as written by your company's setting: {', '.join(a.privacy_kept)}." if a.privacy_kept else ""
            turn.lines.append(_line(f"Personal data found in your message was masked before it reached the model ({kinds}).{kept}"))
        return
    if a.privacy_mode == "off":
        if not sess.said_off:
            sess.said_off = True
            turn.lines.append(_line("Masking is switched off in your company's console setting. "
                                    "Messages go to the model as written."))
        return
    _cannot_mask(turn, s, "Masking was not available for this message")


def _swap_text(text: str, swaps: list, hits: set) -> str:
    for old, new in swaps:
        if old in text:
            text = text.replace(old, new)
            hits.add(old)
    return text


def _map_texts(item: dict, fn: Callable[[str], str]) -> bool:
    """Apply ``fn`` to each text of one message (string content, or each text part). True if any changed."""
    key = "content" if "content" in item else "parts"            # "parts": Gemini
    content, changed = item.get(key), False
    if isinstance(content, str):
        new = fn(content)
        if new != content:
            item[key], changed = new, True
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES and isinstance(part.get("text"), str):
                new = fn(part["text"])
                if new != part["text"]:
                    part["text"], changed = new, True
    return changed


def rewrite_request(session_id: str, request: Any) -> Optional[dict]:
    """Put the masked copy (or a fixed note) into a request about to go to the model.
    Returns the changed request, or None when nothing changed."""
    with _lock:
        sess = _sessions.get(session_id)
    if sess is None or not isinstance(request, dict):
        return None
    turn = sess.turn
    items = next((request[k] for k in ("messages", "input", "contents") if isinstance(request.get(k), list)), [])
    messages = [(ROLES[m["role"]], m) for m in items if isinstance(m, dict) and m.get("role") in ROLES]
    changed = False

    if turn is not None and (turn.blocked or turn.held):
        # Nothing of the conversation goes out on this turn: every user and assistant text becomes the
        # note, and the tools go, so the model gets one short request with nothing to act on.
        for role, m in messages:
            if _map_texts(m, lambda _: turn.expect):
                changed = True
                turn.applied = turn.applied or role == "user"
        for k in ("tools", "tool_choice", "toolConfig"):
            if k in request:
                request.pop(k)
                changed = True
        return request if changed else None

    swaps: dict = {"user": [], "assistant": []}
    for old, note in sess.removed.items():                       # blocked earlier in the session
        swaps["user"].append((old, note))
        swaps["assistant"].append((old, note))
    for role, old, new in turn.swaps if turn is not None else []:
        swaps[role].append((old, new))
    for pairs in swaps.values():
        pairs.sort(key=lambda p: len(p[0]), reverse=True)
    hits: set = set()
    for role, m in messages:
        if swaps[role] and _map_texts(m, lambda t: _swap_text(t, swaps[role], hits)):
            changed = True
    if turn is not None and turn.expect is not None and turn.user_raw in hits:
        turn.applied = True
    return request if changed else None


def finish_turn(session_id: str, response_text: str, settings: Optional[Settings] = None,
                opener: Optional[Callable] = None) -> Optional[str]:
    """The reply the user sees: originals back in, the reply checked going out, EU Mode's lines added.
    Returns None when the reply is unchanged."""
    s = settings or current_settings()
    with _lock:
        sess = _sessions.get(session_id)
    turn = sess.turn if sess else None
    if turn is None or not isinstance(response_text, str):
        return None
    sess.turn = None
    lines = list(turn.lines)
    if turn.expect is not None and not turn.applied:
        # The masked copy (or the note) never made it into a request: never let that pass silently.
        lines.append(_line("EU Mode could not change the request to the model, so your message "
                           "may have reached the model as written."))
    if turn.blocked:
        return "\n\n".join([_blocked_line(turn.check, "this message, so it was not sent to your model")] + lines)
    if turn.held:
        return "\n\n".join(lines)

    reply = restore(response_text, sess.last_map)
    if s.key and turn.check is not None and turn.check.checked:
        a = ask(turn.sent + [("assistant", reply)], s, opener)
        status.update(reply_decision=a.decision if a.checked else None, reply_problem=a.problem)
        if not a.checked:
            log.warning("eu-mode: reply not checked (%s)", a.problem)
            lines.append(_line(f"This reply was not checked: {a.problem}."))
        elif a.decision == "block":
            _remember_blocked(session_id, sess, response_text)
            log.info("eu-mode: model reply blocked (policy %s)", a.policy)
            return "\n\n".join([_blocked_line(a, "the model's reply, so it is not shown")] + lines)
    out = "\n\n".join([reply] + lines)
    return out if out != response_text else None
