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

"""EU Mode (beta) as a Hermes plugin: HaloGuard EU checks every message, and
personal data in what users send is masked before it reaches the model.

Three seams, all in Hermes' published plugin surface (checked against Hermes
0.20.0):

- ``pre_llm_call`` (once per user turn): the check of the conversation that
  ends in the user's message. Its return value is ignored here on purpose:
  EU Mode adds nothing to the model's context.
- ``llm_request`` middleware (every request of the turn, tool loop included):
  the only seam that can change what is sent to the model. It swaps in the
  masked copy, or a fixed note for a blocked message.
- ``transform_llm_output`` (once per turn): originals back into the reply,
  the reply checked going out, EU Mode's lines added.

Everything decided here comes from :mod:`eu_mode_core`; this file only
translates. Each callback catches its own errors: a plugin that raises gets
its exception swallowed by the host, and EU Mode fails open by design. A
masked copy that never reached a request is caught by the core and said in
the reply, so an error here can not hide that masking did not happen.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from . import eu_mode_core as core

logger = logging.getLogger(__name__)


def _sid(session_id: Any) -> str:
    return str(session_id or "default")


# Hermes features that send text to the model through its own side client, which no plugin
# seam reaches (checked in Hermes 0.20.0: agent/auxiliary_client.py has no hook or middleware).
# Both are on unless switched off; EU Mode says so once per session rather than patch Hermes.
SIDE_REQUESTS = (
    (("auxiliary", "title_generation", "enabled"),
     "Hermes' automatic session titles send the first message of each session to the model "
     "without EU Mode (not checked, not masked). To stop that: "
     "hermes config set auxiliary.title_generation.enabled false"),
    (("compression", "enabled"),
     "Hermes' context compression sends earlier messages to the model without EU Mode when a "
     "conversation gets long. To stop that: hermes config set compression.enabled false"),
)


def _is_on(value: Any) -> bool:
    if value is None:
        return True                                   # Hermes' default for both
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _side_lines() -> list:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    except Exception:  # noqa: BLE001 - not inside Hermes, or the module moved
        return []
    out = []
    for path, line in SIDE_REQUESTS:
        node: Any = config
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if _is_on(node):
            out.append(line)
    return out


def _pre_llm_call(session_id: str = "", user_message: str = "", conversation_history: Any = None,
                  **_: Any) -> None:
    try:
        core.start_turn(_sid(session_id), user_message, conversation_history or [], side_lines=_side_lines())
    except Exception as exc:  # noqa: BLE001 - fail open, never break a turn
        logger.warning("eu-mode: check skipped after an error (%s)", type(exc).__name__)
    return None


def _llm_request(request: Any = None, session_id: str = "", **_: Any) -> Optional[dict]:
    try:
        new = core.rewrite_request(_sid(session_id), request)
    except Exception as exc:  # noqa: BLE001
        logger.warning("eu-mode: request not changed after an error (%s)", type(exc).__name__)
        return None
    if new is None:
        return None
    return {"request": new, "source": "eu-mode", "reason": "HaloGuard EU answer applied"}


def _transform_llm_output(response_text: str = "", session_id: str = "", **_: Any) -> Optional[str]:
    try:
        return core.finish_turn(_sid(session_id), response_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("eu-mode: reply not changed after an error (%s)", type(exc).__name__)
        return None


def _forget(session_id: str = "", **_: Any) -> None:
    try:
        core.forget(_sid(session_id))
    except Exception:  # noqa: BLE001
        pass


class _HermesSecrets:
    """EU Mode's settings, read through Hermes's secret scope (``agent.secret_scope.get_secret``),
    which honours the active profile: a bare process-environment read could pick up another
    profile's key when Hermes runs several at once. Anything that goes wrong reads as "not set",
    so EU Mode fails open and says the key is not set."""

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        try:
            from agent.secret_scope import get_secret

            value = get_secret(name, default)
        except Exception:  # noqa: BLE001 - an older Hermes, or no scope while profiles are multiplexed
            return default
        return value if isinstance(value, str) else default


def _cmd_eu_mode(raw_args: str = "") -> str:
    s = core.current_settings()
    st = core.status
    lines = [
        f"EU Mode (beta) {core.VERSION}: HaloGuard EU ({core.MODEL}) at {s.base_url}",
        f"  key: {'set' if s.key else 'NOT set (' + core.KEY_ENV + ')'}",
        f"  if masking is not possible: {'hold the message' if s.require_masking else 'send it as written, and say so'}",
    ]
    if st:
        lines.append(f"  last message: {st.get('decision') or 'not checked'}"
                     + (f" ({st['problem']})" if st.get("problem") else ""))
        if st.get("privacy_mode"):
            lines.append(f"  masking setting: {st['privacy_mode']}; masked: {', '.join(st.get('masked') or []) or 'nothing found'}")
        if "reply_decision" in st:
            lines.append(f"  last reply: {st.get('reply_decision') or 'not checked'}"
                         + (f" ({st['reply_problem']})" if st.get("reply_problem") else ""))
    return "\n".join(lines)


def _store_path() -> Optional[Path]:
    """<Hermes home>/eu-mode/blocked.json: fingerprints of blocked messages, no text."""
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / "eu-mode" / "blocked.json"
    except Exception:  # noqa: BLE001 - not inside Hermes: memory only
        return None


def register(ctx) -> None:
    core.use_env(_HermesSecrets())
    core.use_store(_store_path())
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("transform_llm_output", _transform_llm_output)
    for hook in ("on_session_finalize", "on_session_reset"):
        ctx.register_hook(hook, _forget)
    if hasattr(ctx, "register_middleware"):
        ctx.register_middleware("llm_request", _llm_request)
    else:
        # Older Hermes: no seam to change the request. The guard still runs; every
        # masked turn then says the masked copy did not reach the model.
        logger.warning("eu-mode: this Hermes has no llm_request middleware, so masking cannot be applied")
    try:
        ctx.register_command("eu-mode", _cmd_eu_mode,
                             description="Show EU Mode (beta) status: key set or not, last decision")
    except Exception as exc:  # noqa: BLE001 - the status command is a convenience only
        logger.info("eu-mode: /eu-mode not registered (%s)", type(exc).__name__)
    logger.info("eu-mode %s registered (HaloGuard EU, beta)", core.VERSION)
