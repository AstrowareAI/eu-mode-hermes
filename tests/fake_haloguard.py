"""A fake HaloGuard EU for tests: same request and answer shape as the live API, made-up data only.

It blocks any message containing "SOCIAL SCORE" (policy P2_SOCIAL_SCORING), and masks the made-up
values in VALUES across the whole conversation, numbering placeholders per kind by first appearance,
from 1, as the live filter does (astroware-platform pii/serve.py, mask()). Masking fields come back
only when the last message is a user message, as with the live API.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VALUES = {"Lena Vogt": "PERSON", "Jonas Brandt": "PERSON", "lena.vogt@example.com": "EMAIL",
          "Vogt Logistik GmbH": "ORGANIZATION"}     # all made up


def mask_conversation(texts: list) -> tuple:
    taken: dict = {}
    seen, mapping, out = {}, {}, []
    for t in texts:
        hits = sorted((t.find(v), v) for v in VALUES if v in t)
        for _, v in hits:
            if v not in seen:
                lab = VALUES[v]
                taken[lab] = taken.get(lab, 0) + 1
                seen[v] = f"[{lab}_{taken[lab]}]"
                mapping[seen[v]] = v
        for v in sorted(seen, key=len, reverse=True):
            t = t.replace(v, seen[v])
        out.append(t)
    found = sorted({VALUES[v].lower() for v in seen})
    return out, mapping, found


class FakeHaloGuard:
    def __init__(self):
        self.calls: list = []            # (path, headers dict, body dict)
        self.privacy = "normal"          # normal | unavailable | off | missing
        self.kept: list = []
        self.delay = 0.0
        self.status = 200
        self.error_message = ""
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.calls.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                if fake.delay:
                    time.sleep(fake.delay)
                if fake.status != 200:
                    payload = {"error": {"message": fake.error_message or "error", "type": "x"}}
                    status = fake.status
                else:
                    payload, status = fake.answer(body), 200
                data = json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def answer(self, body: dict) -> dict:
        msgs = body["messages"]
        last = msgs[-1]
        flagged = "SOCIAL SCORE" in last["content"].upper()
        ans = {"decision": "block" if flagged else "allow", "policy": "P2_SOCIAL_SCORING" if flagged else "none"}
        if flagged:
            ans["label"] = "prohibited_practice"
            ans["act_area"] = {"act": "made-up", "reference": "made-up", "qualifier": None}
        if last["role"] == "user" and self.privacy != "missing":
            ans["privacy_mode"] = "off" if self.privacy == "off" else ("custom" if self.kept else "all")
            if self.kept:
                ans["privacy_kept"] = list(self.kept)
            if self.privacy == "unavailable":
                ans["privacy"] = "unavailable"
            elif self.privacy == "normal":
                masked, mapping, found = mask_conversation([m["content"] for m in msgs])
                ans["masked_messages"] = [{"role": m["role"], "content": t} for m, t in zip(msgs, masked)]
                ans["map"], ans["personal_data"] = mapping, found
        return {"id": "hg-test", "object": "chat.completion", "model": "haloguard-eu",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(ans)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


if __name__ == "__main__":       # standalone, for the Hermes end-to-end check: python3 tests/fake_haloguard.py LOGFILE
    import sys

    fake = FakeHaloGuard()
    print(fake.base_url, flush=True)
    try:
        while True:
            time.sleep(0.5)
            with open(sys.argv[1], "w") as f:
                json.dump([{"path": p, "user_agent": h.get("user-agent"), "messages": b["messages"]} for p, h, b in fake.calls], f)
    except KeyboardInterrupt:
        fake.close()
