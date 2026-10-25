"""EU Mode tests against a fake HaloGuard EU on this machine (tests/fake_haloguard.py). Made-up data only.

Run:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import logging
import socket
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eu_mode_core as core  # noqa: E402
from fake_haloguard import FakeHaloGuard  # noqa: E402

KEY = "hg-test-key-made-up-0000"
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}


def chat_request(history: list) -> dict:
    """What Hermes sends to an OpenAI-style model: system prompt, the conversation, tools."""
    return {"model": "some-model", "messages": [copy.deepcopy(SYSTEM)] + copy.deepcopy(history),
            "tools": [{"type": "function", "function": {"name": "web_search"}}]}


class Base(unittest.TestCase):
    def setUp(self):
        self.fake = FakeHaloGuard()
        self.settings = core.Settings(key=KEY, base_url=self.fake.base_url, timeout=2.0)
        core._sessions.clear()
        core.status.clear()
        core.use_store(None)

    def tearDown(self):
        self.fake.close()

    def turn(self, history, user, sid="s1", settings=None):
        """One user turn up to the model request: returns (history incl. the user message, request sent)."""
        history = history + [{"role": "user", "content": user}]
        core.start_turn(sid, user, history, settings or self.settings)
        request = chat_request(history)
        new = core.rewrite_request(sid, request)
        return history, (new if new is not None else request)

    def users(self, request):
        return [m["content"] for m in request["messages"] if m["role"] == "user"]


class Allow(Base):
    def test_allow_leaves_everything_as_is_and_checks_both_ways(self):
        h, req = self.turn([], "What is the capital of France?")
        self.assertEqual(self.users(req), ["What is the capital of France?"])
        self.assertIn("tools", req)
        self.assertIsNone(core.finish_turn("s1", "Paris.", self.settings))
        self.assertEqual(len(self.fake.calls), 2)
        (p1, h1, b1), (p2, _, b2) = self.fake.calls
        self.assertEqual(p1, "/v1/chat/completions")
        self.assertEqual(h1["user-agent"], "eu-mode-hermes/0.1")
        self.assertEqual(h1["authorization"], "Bearer " + KEY)
        self.assertEqual(b1["model"], "haloguard-eu")
        self.assertEqual(b1["messages"], [{"role": "user", "content": "What is the capital of France?"}])
        self.assertEqual(b2["messages"][-1], {"role": "assistant", "content": "Paris."})

    def test_system_prompt_and_tool_results_are_never_sent_to_haloguard(self):
        history = [{"role": "user", "content": "Find the weather."},
                   {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                   {"role": "tool", "content": "Sunny, made-up tool output", "tool_call_id": "1"},
                   {"role": "assistant", "content": "It is sunny."}]
        self.turn(history, "Thanks!")
        sent = self.fake.calls[0][2]["messages"]
        self.assertEqual([m["role"] for m in sent], ["user", "assistant", "user"])
        self.assertNotIn("made-up tool output", json.dumps(sent))


class Block(Base):
    def test_blocked_message_never_reaches_the_model(self):
        msg = "Give each tenant a SOCIAL SCORE from their posts and evict the lowest."
        h, req = self.turn([], msg)
        self.assertEqual(self.users(req), [core.REMOVED_NOTE])
        self.assertNotIn(msg, json.dumps(req))
        self.assertNotIn("tools", req)
        out = core.finish_turn("s1", "OK.", self.settings)
        self.assertIn("HaloGuard EU blocked this message, so it was not sent to your model", out)
        self.assertIn("P2_SOCIAL_SCORING (prohibited_practice)", out)
        self.assertEqual(len(self.fake.calls), 1)          # no reply check for a blocked message

        # Later turns: the blocked text stays out of both the check and the model request.
        h2, req2 = self.turn(h + [{"role": "assistant", "content": "OK."}], "What can you help with?")
        self.assertNotIn(msg, json.dumps(self.fake.calls[1][2]))
        self.assertNotIn(msg, json.dumps(req2))
        self.assertIn("tools", req2)

    def test_a_blocked_turn_sends_nothing_of_the_conversation(self):
        h, _ = self.turn([], "Write to Lena Vogt about Friday.")
        core.finish_turn("s1", "Done for [PERSON_1].", self.settings)
        h = h + [{"role": "assistant", "content": "Done for [PERSON_1]."}]
        _, req = self.turn(h, "Now give her a SOCIAL SCORE.")
        self.assertEqual({m["content"] for m in req["messages"] if m["role"] != "system"}, {core.REMOVED_NOTE})
        self.assertNotIn("Lena Vogt", json.dumps(req))
        self.assertEqual(req["messages"][0], SYSTEM)

    def test_a_block_holds_after_a_restart_and_only_a_fingerprint_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            store = Path(d) / "eu-mode" / "blocked.json"
            core.use_store(store)
            msg = "Give each tenant a SOCIAL SCORE."
            h, _ = self.turn([], msg)
            core.finish_turn("s1", "OK.", self.settings)
            core._sessions.clear()                                  # Hermes restarts
            h = h + [{"role": "assistant", "content": "OK."}]
            _, req = self.turn(h, "What else can you do?")
            self.assertNotIn(msg, json.dumps(req))
            self.assertNotIn(msg, json.dumps(self.fake.calls[-1][2]))
            raw = store.read_text()
            self.assertNotIn("SOCIAL", raw)
            self.assertIn(hashlib.sha256(msg.encode()).hexdigest(), raw)
            self.assertEqual(stat.S_IMODE(store.stat().st_mode), 0o600)

    def test_reply_going_out_is_checked_and_a_blocked_reply_is_not_shown(self):
        h, req = self.turn([], "Summarise our tenant policy.")
        out = core.finish_turn("s1", "Step one: compute a SOCIAL SCORE for every tenant.", self.settings)
        self.assertIn("HaloGuard EU blocked the model's reply, so it is not shown", out)
        self.assertNotIn("Step one", out)
        reply_check = self.fake.calls[1][2]["messages"]
        self.assertEqual(reply_check[-1]["role"], "assistant")
        # Next turn: the withheld reply is not sent back to the model as if the user had seen it.
        h2 = h + [{"role": "assistant", "content": "Step one: compute a SOCIAL SCORE for every tenant."}]
        _, req2 = self.turn(h2, "Go on.")
        self.assertNotIn("Step one", json.dumps(req2))


class Masking(Base):
    def test_masked_request_and_originals_back_in_the_reply(self):
        msg = "Write a short note to Lena Vogt (lena.vogt@example.com) about Friday."
        h, req = self.turn([], msg)
        self.assertEqual(self.users(req), ["Write a short note to [PERSON_1] ([EMAIL_1]) about Friday."])
        self.assertNotIn("Lena Vogt", json.dumps(req))
        self.assertEqual(req["messages"][0], SYSTEM)       # the system prompt is not touched
        out = core.finish_turn("s1", "Dear [PERSON_1], see you Friday.", self.settings)
        self.assertTrue(out.startswith("Dear Lena Vogt, see you Friday."))
        self.assertIn("Personal data found in your message was masked before it reached the model (email, person).", out)
        # The reply is checked as the user sees it, and the reply check carries no masking.
        self.assertEqual(self.fake.calls[1][2]["messages"][-1]["content"], "Dear Lena Vogt, see you Friday.")

    def test_every_request_of_the_turn_is_masked_tool_results_are_not(self):
        msg = "Look up Lena Vogt's office hours."
        h = [{"role": "user", "content": msg}]
        core.start_turn("s1", msg, h, self.settings)
        for extra in ([], [{"role": "assistant", "content": "Searching.", "tool_calls": [{"id": "1"}]},
                           {"role": "tool", "content": "Lena Vogt: 9-12 (made-up)", "tool_call_id": "1"}]):
            req = chat_request(h + extra)
            core.rewrite_request("s1", req)
            self.assertEqual(self.users(req), ["Look up [PERSON_1]'s office hours."])
        tool = [m for m in req["messages"] if m["role"] == "tool"][0]
        self.assertEqual(tool["content"], "Lena Vogt: 9-12 (made-up)")     # only what users send is masked

    def test_one_set_of_placeholders_across_the_conversation(self):
        h, _ = self.turn([], "Email Lena Vogt the agenda.")
        core.finish_turn("s1", "Done: emailed [PERSON_1].", self.settings)
        h = h + [{"role": "assistant", "content": "Done: emailed [PERSON_1]."}]     # Hermes keeps the raw reply
        _, req = self.turn(h, "Now copy Jonas Brandt.")
        sent = self.fake.calls[2][2]["messages"]
        self.assertEqual(sent[1]["content"], "Done: emailed Lena Vogt.")          # originals back before the check
        self.assertEqual([m["content"] for m in req["messages"][1:]],
                         ["Email [PERSON_1] the agenda.", "Done: emailed [PERSON_1].", "Now copy [PERSON_2]."])

    def test_kept_kinds_are_named(self):
        self.fake.kept = ["names"]
        self.turn([], "Ask lena.vogt@example.com for the file.")
        out = core.finish_turn("s1", "Asked [EMAIL_1].", self.settings)
        self.assertIn("Kept as written by your company's setting: names.", out)

    def test_host_side_requests_are_said_once_per_session(self):
        side = ["Session titles go to the model without EU Mode."]
        msg = "Hello there."
        core.start_turn("s1", msg, [{"role": "user", "content": msg}], self.settings, side_lines=side)
        self.assertIn("[EU Mode beta] Session titles go to the model without EU Mode.", core.finish_turn("s1", "Hi.", self.settings))
        h = [{"role": "user", "content": msg}, {"role": "assistant", "content": "Hi."}, {"role": "user", "content": "Bye."}]
        core.start_turn("s1", "Bye.", h, self.settings, side_lines=side)
        self.assertIsNone(core.finish_turn("s1", "Bye.", self.settings))

    def test_masked_copy_that_never_reached_a_request_is_said(self):
        core.start_turn("s1", "Call Lena Vogt.", [{"role": "user", "content": "Call Lena Vogt."}], self.settings)
        out = core.finish_turn("s1", "Calling.", self.settings)      # no llm_request ran in between
        self.assertIn("may have reached the model as written", out)


class PrivacyUnavailable(Base):
    def test_fails_open_and_says_so(self):
        self.fake.privacy = "unavailable"
        h, req = self.turn([], "Call Lena Vogt.")
        self.assertEqual(self.users(req), ["Call Lena Vogt."])
        out = core.finish_turn("s1", "Calling Lena Vogt.", self.settings)
        self.assertIn("Masking was not available for this message. Your message went to the model as written, not masked.", out)

    def test_require_masking_holds_the_message(self):
        self.fake.privacy = "unavailable"
        strict = core.Settings(key=KEY, base_url=self.fake.base_url, timeout=2.0, require_masking=True)
        h, req = self.turn([], "Call Lena Vogt.", settings=strict)
        self.assertEqual(self.users(req), [core.HELD_NOTE])
        self.assertNotIn("tools", req)
        out = core.finish_turn("s1", "OK.", strict)
        self.assertIn("so your message was NOT sent to the model (EU_MODE_REQUIRE_MASKING is on)", out)

    def test_a_held_turn_sends_nothing_of_the_conversation(self):
        h, _ = self.turn([], "Write to Lena Vogt about Friday.")
        core.finish_turn("s1", "Done for [PERSON_1].", self.settings)
        h = h + [{"role": "assistant", "content": "Done for [PERSON_1]."}]
        self.fake.privacy = "unavailable"
        strict = core.Settings(key=KEY, base_url=self.fake.base_url, timeout=2.0, require_masking=True)
        _, req = self.turn(h, "And Jonas Brandt too.", settings=strict)
        self.assertEqual({m["content"] for m in req["messages"] if m["role"] != "system"}, {core.HELD_NOTE})
        self.assertNotIn("Lena Vogt", json.dumps(req))

    def test_masking_switched_off_by_the_company_is_said_once(self):
        self.fake.privacy = "off"
        h, _ = self.turn([], "Call Lena Vogt.")
        self.assertIn("Masking is switched off", core.finish_turn("s1", "Calling.", self.settings))
        h = h + [{"role": "assistant", "content": "Calling."}]
        self.turn(h, "Thanks.")
        self.assertIsNone(core.finish_turn("s1", "You're welcome.", self.settings))

    def test_api_without_masking_fields_is_treated_as_unavailable(self):
        self.fake.privacy = "missing"
        self.turn([], "Call Lena Vogt.")
        self.assertIn("Masking was not available", core.finish_turn("s1", "Calling.", self.settings))


class FailsOpen(Base):
    def test_timeout_fails_open_quickly(self):
        self.fake.delay = 3.0
        quick = core.Settings(key=KEY, base_url=self.fake.base_url, timeout=0.5)
        t0 = time.monotonic()
        h, req = self.turn([], "Call Lena Vogt.", settings=quick)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(self.users(req), ["Call Lena Vogt."])      # the turn goes on
        self.assertIn("tools", req)
        out = core.finish_turn("s1", "Calling.", quick)
        self.assertIn("HaloGuard EU did not answer in time. Your message went to the model unchecked and not masked.", out)
        self.assertEqual(len(self.fake.calls), 1)                   # no second wait for the reply check

    def test_unreachable_fails_open(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        gone = core.Settings(key=KEY, base_url=f"http://127.0.0.1:{port}/v1", timeout=1.0)
        self.turn([], "Hello", settings=gone)
        self.assertIn("could not be reached", core.finish_turn("s1", "Hi.", gone))

    def test_bad_key_fails_open(self):
        self.fake.status = 401
        self.turn([], "Hello")
        self.assertIn("did not accept the key (HTTP 401)", core.finish_turn("s1", "Hi.", self.settings))

    def test_budget_message_is_shown(self):
        self.fake.status, self.fake.error_message = 400, "Budget has been exceeded (made-up)"
        self.turn([], "Hello")
        self.assertIn("HTTP 400: Budget has been exceeded (made-up)", core.finish_turn("s1", "Hi.", self.settings))

    def test_unreadable_answer_fails_open(self):
        self.fake.answer = lambda body: {"choices": [{"message": {"content": "not json"}}]}
        self.turn([], "Hello")
        self.assertIn("could not read", core.finish_turn("s1", "Hi.", self.settings))

    def test_no_key_makes_no_call_and_says_so(self):
        nokey = core.Settings(key="", base_url=self.fake.base_url)
        h, req = self.turn([], "Call Lena Vogt.", settings=nokey)
        self.assertEqual(self.users(req), ["Call Lena Vogt."])
        out = core.finish_turn("s1", "Calling.", nokey)
        self.assertIn("HALOGUARD_API_KEY is not set. Nothing was checked or masked.", out)
        self.assertEqual(self.fake.calls, [])

    def test_key_never_goes_over_plain_http_to_another_host(self):
        bad = core.Settings(key=KEY, base_url="http://example.com/v1")
        self.turn([], "Hello", settings=bad)
        self.assertIn("must be an https:// address", core.finish_turn("s1", "Hi.", bad))
        self.assertEqual(self.fake.calls, [])


class KeyStaysSecret(Base):
    def test_key_is_not_in_logs_reprs_or_replies(self):
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        root = logging.getLogger()
        root.addHandler(handler)
        old = root.level
        root.setLevel(logging.DEBUG)
        try:
            outs = []
            for privacy, status in (("normal", 200), ("unavailable", 200), ("normal", 401), ("normal", 500)):
                core._sessions.clear()
                self.fake.privacy, self.fake.status = privacy, status
                self.turn([], "Call Lena Vogt about the SOCIAL SCORE plan.")
                outs.append(core.finish_turn("s1", "Calling.", self.settings) or "")
            self.assertNotIn(KEY, buf.getvalue())
            self.assertNotIn(KEY, repr(self.settings))
            self.assertNotIn(KEY, "".join(outs))
            self.assertNotIn(KEY, json.dumps(core.status))
            self.assertNotIn("Lena Vogt", buf.getvalue())       # no message text in logs either
        finally:
            root.removeHandler(handler)
            root.setLevel(old)

    def test_settings_from_env(self):
        s = core.Settings.from_env({"HALOGUARD_API_KEY": " k ", "EU_MODE_TIMEOUT": "999", "EU_MODE_REQUIRE_MASKING": "yes"})
        self.assertEqual((s.key, s.timeout, s.require_masking, s.base_url, s.footer),
                         ("k", 30.0, True, "https://api.astroware.ai/v1", True))
        self.assertEqual(core.Settings.from_env({"EU_MODE_TIMEOUT": "x"}).timeout, core.DEFAULT_TIMEOUT)


class RequestShapes(Base):
    def start(self, text):
        core.start_turn("s1", text, [{"role": "user", "content": text}], self.settings)

    def test_anthropic_blocks(self):
        self.start("Call Lena Vogt.")
        req = {"system": "sys", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Call Lena Vogt."},
            {"type": "tool_result", "tool_use_id": "1", "content": "Lena Vogt (tool output)"}]}]}
        core.rewrite_request("s1", req)
        blocks = req["messages"][0]["content"]
        self.assertEqual(blocks[0]["text"], "Call [PERSON_1].")
        self.assertEqual(blocks[1]["content"], "Lena Vogt (tool output)")

    def test_codex_responses_input(self):
        self.start("Call Lena Vogt.")
        req = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Call Lena Vogt."}]},
                         {"type": "function_call_output", "output": "Lena Vogt"}]}
        core.rewrite_request("s1", req)
        self.assertEqual(req["input"][0]["content"][0]["text"], "Call [PERSON_1].")

    def test_bedrock_converse_parts(self):
        self.start("Call Lena Vogt.")
        req = {"messages": [{"role": "user", "content": [{"text": "Call Lena Vogt."}]}], "toolConfig": {}}
        core.rewrite_request("s1", req)
        self.assertEqual(req["messages"][0]["content"][0]["text"], "Call [PERSON_1].")

    def test_gemini_contents(self):
        self.start("Call Lena Vogt.")
        req = {"contents": [{"role": "user", "parts": [{"text": "Call Lena Vogt."}]}]}
        core.rewrite_request("s1", req)
        self.assertEqual(req["contents"][0]["parts"][0]["text"], "Call [PERSON_1].")

    def test_extra_context_around_the_user_message_is_kept(self):
        self.start("Call Lena Vogt.")
        req = {"messages": [{"role": "user", "content": "Call Lena Vogt.\n\n[memory note from another plugin]"}]}
        core.rewrite_request("s1", req)
        self.assertEqual(req["messages"][0]["content"], "Call [PERSON_1].\n\n[memory note from another plugin]")


class FakeCtx:
    def __init__(self):
        self.hooks, self.middleware, self.commands = {}, {}, {}

    def register_hook(self, name, cb):
        self.hooks.setdefault(name, []).append(cb)

    def register_middleware(self, kind, cb):
        self.middleware.setdefault(kind, []).append(cb)

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = handler


def load_plugin():
    """A fresh copy of the plugin package, loaded the way Hermes loads it (relative imports work)."""
    for name in ("eu_mode_plugin", "eu_mode_plugin.eu_mode_core"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location("eu_mode_plugin", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eu_mode_plugin"] = mod
    spec.loader.exec_module(mod)
    return mod


class HermesGlue(Base):
    def test_register_and_one_turn_through_hermes_shaped_calls(self):
        plugin = load_plugin()
        ctx = FakeCtx()
        plugin.register(ctx)
        self.assertEqual(sorted(ctx.hooks), ["on_session_finalize", "on_session_reset", "pre_llm_call", "transform_llm_output"])
        self.assertIn("llm_request", ctx.middleware)
        self.assertIn("eu-mode", ctx.commands)
        self.assertNotIn("pre_tool_call", ctx.hooks)          # no tool-call gating in EU Mode

        env = {"HALOGUARD_API_KEY": KEY, "HALOGUARD_BASE_URL": self.fake.base_url}
        orig = plugin.core.Settings.from_env
        plugin.core.Settings.from_env = classmethod(lambda cls, e=None: orig(env))
        try:
            msg = "Write to Lena Vogt."
            history = [{"role": "user", "content": msg}]
            self.assertIsNone(ctx.hooks["pre_llm_call"][0](session_id="abc", user_message=msg,
                                                           conversation_history=history, is_first_turn=True))
            res = ctx.middleware["llm_request"][0](request=chat_request(history), session_id="abc",
                                                   api_mode="chat_completions", telemetry_schema_version="x")
            self.assertEqual(res["request"]["messages"][1]["content"], "Write to [PERSON_1].")
            out = ctx.hooks["transform_llm_output"][0](response_text="Hi [PERSON_1]!", session_id="abc", model="m")
            self.assertTrue(out.startswith("Hi Lena Vogt!"))
            status = ctx.commands["eu-mode"]("")
            self.assertIn("key: set", status)
            self.assertNotIn(KEY, status)
        finally:
            plugin.core.Settings.from_env = orig

    def test_side_requests_are_named_from_hermes_config(self):
        plugin = load_plugin()
        fake_config = type(sys)("hermes_cli.config")
        fake_config.load_config_readonly = lambda: {"auxiliary": {"title_generation": {"enabled": False}},
                                                    "compression": {"enabled": "true"}}
        fake_pkg = type(sys)("hermes_cli")
        saved = {k: sys.modules.get(k) for k in ("hermes_cli", "hermes_cli.config")}
        sys.modules.update({"hermes_cli": fake_pkg, "hermes_cli.config": fake_config})
        try:
            lines = plugin._side_lines()
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self.assertEqual(len(lines), 1)
        self.assertIn("hermes config set compression.enabled false", lines[0])

    def test_errors_never_break_a_turn(self):
        plugin = load_plugin()
        boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))   # noqa: E731
        for name in ("start_turn", "rewrite_request", "finish_turn"):
            setattr(plugin.core, name, boom)
        self.assertIsNone(plugin._pre_llm_call(session_id="x", user_message="hi", conversation_history=[]))
        self.assertIsNone(plugin._llm_request(request={"messages": []}, session_id="x"))
        self.assertIsNone(plugin._transform_llm_output(response_text="hi", session_id="x"))


if __name__ == "__main__":
    unittest.main()
