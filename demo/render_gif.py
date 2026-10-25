"""Draw a terminal-style GIF from a list of steps, with Pillow only (no recorder to install).

Steps: {"type": text} types a command; {"print": text, "style": name} shows a line;
{"hold": seconds} waits. Every text shown comes from the caller: this file draws, it invents nothing.
"""

from __future__ import annotations

import textwrap

from PIL import Image, ImageDraw, ImageFont

W, H, MARGIN, TOP = 960, 360, 26, 58
FONT_FILE = "/System/Library/Fonts/Menlo.ttc"
BG = (15, 17, 23)
STYLES = {
    "cmd": (230, 230, 230),
    "prompt": (126, 231, 135),
    "reply": (230, 230, 230),
    "model": (210, 168, 255),
    "eu": (121, 192, 255),
    "block": (255, 166, 87),
    "dim": (139, 148, 158),
}


class Screen:
    def __init__(self, title: str):
        self.font = ImageFont.truetype(FONT_FILE, 16)
        self.title_font = ImageFont.truetype(FONT_FILE, 15)
        self.cw = self.font.getlength("M")
        self.lh = 23
        self.cols = int((W - 2 * MARGIN) / self.cw)
        self.rows = (H - TOP - MARGIN) // self.lh
        self.title = title
        self.lines: list = []          # [(segments [(text, style)])]
        self.frames: list = []
        self.durations: list = []

    def _draw(self) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            d.ellipse((MARGIN + i * 20, 20, MARGIN + i * 20 + 11, 31), fill=c)
        d.text((MARGIN + 72, 17), self.title, font=self.title_font, fill=STYLES["dim"])
        for row, segments in enumerate(self.lines[-self.rows:]):
            x, y = MARGIN, TOP + row * self.lh
            for text, style in segments:
                d.text((x, y), text, font=self.font, fill=STYLES[style])
                x += self.cw * len(text)
        return img.convert("P", palette=Image.Palette.ADAPTIVE, colors=64)

    def snap(self, seconds: float) -> None:
        self.frames.append(self._draw())
        self.durations.append(int(seconds * 1000))

    def print(self, text: str, style: str) -> None:
        for raw in text.split("\n"):
            wrapped = textwrap.wrap(raw, self.cols, subsequent_indent="  ") or [""]
            self.lines.extend([[(w, style)] for w in wrapped])

    def type(self, command: str) -> None:
        start = len(self.lines)
        for end in list(range(3, len(command), 3)) + [len(command)]:
            del self.lines[start:]
            for i, w in enumerate(textwrap.wrap(command[:end], self.cols - 2) or [""]):
                self.lines.append([("$ " if i == 0 else "  ", "prompt"), (w, "cmd")])
            self.snap(0.04)


def render(title: str, steps: list, out: str) -> None:
    s = Screen(title)
    s.snap(1.0)
    for step in steps:
        if "type" in step:
            s.type(step["type"])
        elif "print" in step:
            s.print(step["print"], step.get("style", "reply"))
            s.snap(step.get("after", 0.5))
        elif "hold" in step:
            s.snap(step["hold"])
    s.frames[0].save(out, save_all=True, append_images=s.frames[1:], duration=s.durations, loop=0, optimize=True)
