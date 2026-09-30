"""Renders short on-screen text (facts/figures) as a transparent PNG overlay
using Pillow, since this ffmpeg build has no libass/freetype for drawtext.
"""
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Arial Bold's ₹ glyph renders as tofu (missing-glyph box) on macOS — Helvetica.ttc
# (bold face, index 1) is the confirmed-working macOS font. On the Linux container
# (apt package fonts-noto-core), Noto Sans Bold covers the rupee glyph instead.
FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", 0),
    ("/System/Library/Fonts/Helvetica.ttc", 1),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path, index in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=index)
    return ImageFont.load_default()


def render_text_card(text: str, width: int, height: int, out_path: Path) -> Path:
    """Renders `text` inside a semi-transparent bar near the bottom third of a
    `width`x`height` transparent canvas, sized for the video's own resolution."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_size = max(28, width // 18)
    font = _load_font(font_size)

    max_chars_per_line = max(10, width // (font_size // 2))
    lines = textwrap.wrap(text, width=max_chars_per_line) or [text]

    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    line_height = max(line_heights) if line_heights else font_size
    padding = int(font_size * 0.6)
    bar_height = padding * 2 + line_height * len(lines) + (len(lines) - 1) * int(font_size * 0.25)

    bar_y0 = int(height * 0.68)
    bar_y1 = min(height, bar_y0 + bar_height)
    draw.rectangle([0, bar_y0, width, bar_y1], fill=(0, 0, 0, 170))

    y = bar_y0 + padding
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (width - text_w) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height + int(font_size * 0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
