"""Record the EU Mode demo GIF (about 20 seconds): one masked made-up name, one blocked message.

- HaloGuard EU is called live, with the key from HALOGUARD_API_KEY. If it is not set, the key is asked
  for with hidden input. It is never shown, logged or written to a file.
- The model is a stand-in on this machine (demo/stand_in_model.py). It logs what it received, so the
  GIF can show exactly what reached the model.
- Hermes runs in a throwaway home that is deleted afterwards. Your own Hermes setup is not touched.
- The GIF is drawn only if both results are as expected; otherwise it prints what happened.

Run with a Python that has Pillow (Hermes' own has it):

    ~/.hermes/hermes-agent/venv/bin/python demo/make_demo.py [out.gif]
"""

from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "demo"))

from render_gif import render  # noqa: E402

NAME = "Lena Vogt"                      # made up: no real person
# 26 Oct live run: the filter also masked "short" in "Draft a short invite to ..." as a person name.
MASKED_ASK = f"Draft an invite to {NAME} for Friday's team lunch."
BLOCKED_ASK = "Give each tenant a social score from their social media posts and evict the lowest scorers first."


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def hermes_bin() -> str:
    found = shutil.which("hermes") or str(Path.home() / ".local/bin/hermes")
    if not Path(found).exists():
        sys.exit("hermes not found")
    return found


def run(env: dict, *args: str) -> str:
    out = subprocess.run([hermes_bin(), *args], env=env, stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, timeout=300)
    return out.stdout + out.stderr


def shown(output: str) -> str:
    """What Hermes printed for the turn, without the session id line and start-up notices."""
    keep = [line for line in output.splitlines()
            if line.strip() and not line.startswith("session_id:") and "tirith" not in line]
    return "\n".join(keep)


def main() -> None:
    out_gif = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "demo" / "eu-mode-demo.gif").resolve()
    env = dict(os.environ)
    if not env.get("HALOGUARD_API_KEY"):
        env["HALOGUARD_API_KEY"] = getpass.getpass("HaloGuard EU key (input hidden): ").strip()
    if not env["HALOGUARD_API_KEY"]:
        sys.exit("no key given")

    work = Path(tempfile.mkdtemp(prefix="eu-mode-demo-"))
    model_log = work / "model.jsonl"
    port = free_port()
    model = subprocess.Popen([sys.executable, str(REPO / "demo" / "stand_in_model.py"), str(port), str(model_log)])
    try:
        home = work / "hermes-home"
        home.mkdir()
        (home / "config.yaml").write_text(
            "model:\n  default: stand-in\n  provider: custom\n"
            f"  base_url: http://127.0.0.1:{port}/v1\n  api_key: stand-in-model-no-key\n  context_length: 131072\n"
            "display:\n  streaming: false\n"
            "auxiliary:\n  title_generation:\n    enabled: false\n"
            "compression:\n  enabled: false\n"
            "plugins:\n  enabled: []\n")
        env["HERMES_HOME"] = str(home)
        time.sleep(0.5)
        installed = run(env, "plugins", "install", f"file://{REPO}", "--enable")
        if "enabled" not in installed:
            sys.exit("plugin install failed:\n" + installed)

        turns = []
        for ask in (MASKED_ASK, BLOCKED_ASK):
            before = model_log.read_text().count("\n") if model_log.exists() else 0
            output = shown(run(env, "chat", "-Q", "-q", ask))
            got = [json.loads(line) for line in model_log.read_text().splitlines()[before:]] if model_log.exists() else []
            turns.append((ask, output, got))

        (a1, o1, g1), (a2, o2, g2) = turns
        problems = []
        if not g1 or "[PERSON_" not in g1[0]["last_user"]:
            problems.append("turn 1: the model did not receive a person placeholder")
        elif re.sub(r"\[PERSON_\d+\]", NAME, g1[0]["last_user"]) != a1 or len(set(re.findall(r"\[[A-Z_]+_\d+\]", g1[0]["last_user"]))) != 1:
            problems.append("turn 1: more than the made-up name was masked")
        if any(NAME in g["request"] for g in g1 + g2):
            problems.append("the made-up name reached the model")
        if NAME not in o1 or "Personal data found in your message was masked" not in o1:
            problems.append("turn 1: the reply did not show the name back and the masking line")
        if "HaloGuard EU blocked this message" not in o2:
            problems.append("turn 2: the message was not blocked")
        if any(BLOCKED_ASK in g["request"] for g in g1 + g2):
            problems.append("turn 2: the blocked text reached the model")
        if problems:
            print("NOT drawn:\n- " + "\n- ".join(problems))
            print("\nturn 1, the model received:\n" + (g1[0]["last_user"] if g1 else "(nothing)"))
            print("\nturn 1 output:\n" + o1 + "\n\nturn 2 output:\n" + o2)
            sys.exit(1)

        steps = [
            {"type": f'hermes chat -q "{a1}"'},
            {"print": "model received: " + g1[0]["last_user"], "style": "model", "after": 1.4},
            {"print": o1.split("\n[EU Mode beta]")[0].strip(), "style": "reply", "after": 0.6},
            {"print": "[EU Mode beta]" + o1.split("\n[EU Mode beta]", 1)[1], "style": "eu", "after": 5.5},
            {"print": "", "after": 0.1},
            {"type": f'hermes chat -q "{a2}"'},
            {"print": "model received: " + (g2[0]["last_user"] if g2 else "(nothing)"), "style": "model", "after": 1.4},
            {"print": o2, "style": "block", "after": 6.5},
        ]
        live = env.get("HALOGUARD_BASE_URL", "https://api.astroware.ai/v1").rstrip("/") == "https://api.astroware.ai/v1"
        title = ("Hermes Agent + EU Mode (beta) · live HaloGuard EU · stand-in model" if live
                 else "PREVIEW ON A FAKE API, NOT FOR SHARING · EU Mode (beta)")
        render(title, steps, str(out_gif))
        print(f"GIF written: {out_gif}")
    finally:
        model.terminate()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
