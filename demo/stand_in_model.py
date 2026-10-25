"""A stand-in model for the demo GIF: OpenAI-compatible, on this machine. It writes the last user
message of every request it gets to a log, so the GIF can show exactly what reached the model, and
answers with a short fixed invite addressed to the first person placeholder it received.

    python3 demo/stand_in_model.py PORT LOGFILE
"""

from __future__ import annotations

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def reply_to(last_user: str) -> str:
    person = re.search(r"\[PERSON_\d+\]", last_user)
    if person:
        return (f"Hi {person.group(0)}, we're having a team lunch this Friday at 12:30. "
                "Would you like to join us?")
    return "OK."


def main(port: int, logfile: str) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._json({"object": "list", "data": [{"id": "stand-in", "object": "model", "context_length": 131072}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            users = [text_of(m.get("content")) for m in body.get("messages", []) if m.get("role") == "user"]
            last = users[-1].split("\n\n")[0] if users else ""
            with open(logfile, "a", encoding="utf-8") as f:
                f.write(json.dumps({"last_user": last, "has_tools": "tools" in body,
                                    "request": json.dumps(body)}) + "\n")
            reply = reply_to(last)
            usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            if not body.get("stream"):
                self._json({"id": "stand-in-1", "object": "chat.completion", "created": int(time.time()),
                            "model": "stand-in", "usage": usage,
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                                         "finish_reason": "stop"}]})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            base = {"id": "stand-in-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": "stand-in"}
            for delta, finish in (({"role": "assistant", "content": reply}, None), ({}, "stop")):
                self.wfile.write(f"data: {json.dumps(dict(base, choices=[{'index': 0, 'delta': delta, 'finish_reason': finish}]))}\n\n".encode())
            self.wfile.write(f"data: {json.dumps(dict(base, choices=[], usage=usage))}\n\ndata: [DONE]\n\n".encode())

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
