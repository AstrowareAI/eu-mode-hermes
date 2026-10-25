"""A fake OpenAI-compatible model for the Hermes end-to-end check: it answers "You said: <last user text>"
and appends every request it gets to a JSON-lines file, so the check can see exactly what reached the model.
With "tool" as third argument it first calls the terminal tool to "search" for the user's text (a harmless
echo), then answers with what the tool returned: this shows what a research agent's tools receive.

    python3 tests/fake_model.py PORT LOGFILE [tool]
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def main(port: int, logfile: str, tool_mode: bool = False) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._json(200, {"object": "list", "data": [{"id": "fake-model", "object": "model", "context_length": 32768}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with open(logfile, "a") as f:
                f.write(json.dumps(body) + "\n")
            msgs = body.get("messages", [])
            users = [text_of(m.get("content")) for m in msgs if m.get("role") == "user"]
            reply = "You said: " + (users[-1].split("\n\n")[0] if users else "")
            usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            tools = {t.get("function", {}).get("name") for t in body.get("tools") or []}
            message = {"role": "assistant", "content": reply}
            if tool_mode and msgs and msgs[-1].get("role") == "tool":
                message["content"] = "Search result: " + text_of(msgs[-1].get("content"))[:300]
            elif tool_mode and "terminal" in tools and msgs and msgs[-1].get("role") == "user":
                query = users[-1].split("\n\n")[0].replace("'", "")
                message = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "terminal", "arguments": json.dumps({"command": f"echo 'web search for: {query}'"})}}]}
            if not body.get("stream"):
                self._json(200, {"id": "fake-1", "object": "chat.completion", "created": int(time.time()), "model": "fake-model",
                                 "choices": [{"index": 0, "message": message,
                                              "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}], "usage": usage})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            base = {"id": "fake-1", "object": "chat.completion.chunk", "created": int(time.time()), "model": "fake-model"}
            if "tool_calls" in message:
                tc = dict(message["tool_calls"][0], index=0)
                steps = (({"role": "assistant", "tool_calls": [tc]}, None), ({}, "tool_calls"))
            else:
                steps = (({"role": "assistant", "content": message["content"]}, None), ({}, "stop"))
            for delta, finish in steps:
                chunk = dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": finish}])
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(f"data: {json.dumps(dict(base, choices=[], usage=usage))}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "tool")
