"""Renders short on-screen text (facts/figures) as a transparent PNG overlay
using Pillow, since this ffmpeg build has no libass/freetype for drawtext.
"""
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

# Tamil/Kannada/Telugu/Malayalam/Devanagari reorder certain vowel signs before
# their base consonant — plain freetype rendering (Pillow's default layout
# engine) draws those as a disconnected dotted-circle + glyph instead of the
# correctly shaped letter. RAQM (harfbuzz+fribidi) fixes this, but it's an
# opt-in Pillow build (see Dockerfile, which compiles Pillow from source
# against libraqm-dev) — PyPI's prebuilt wheel doesn't include it, so this
# falls back to Pillow's default layout wherever raqm isn't present (e.g.
# local macOS dev) rather than crashing.
_RAQM_AVAILABLE = features.check("raqm")

# Arial Bold's ₹ glyph renders as tofu (missing-glyph box) on macOS — Helvetica.ttc
# (bold face, index 1) is the confirmed-working macOS font. On the Linux container
# (apt package fonts-noto-core), Noto Sans Bold covers the rupee glyph instead.
# Used for Latin-script text (and as the final fallback for any script).
FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", 0),
    ("/System/Library/Fonts/Helvetica.ttc", 1),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
]

# Non-Latin on-screen text (e.g. a Hindi/Tamil story transcript narrated with a
# matching /voice) needs a font that actually has those glyphs — Noto Sans Bold
# above only covers Latin/Cyrillic/Greek. Each entry is (unicode codepoint
# range, Linux font candidates, macOS font candidates) — installed via
# fonts-noto-extra in the Docker image (see Dockerfile); on macOS we fall back
# to whatever Apple system font covers that script, if any.
SCRIPT_FONT_CANDIDATES = [
    ((0x0900, 0x097F), [  # Devanagari (Hindi, Marathi)
        ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf", 0),
        ("/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc", 0),
    ]),
    ((0x0980, 0x09FF), [  # Bengali
        ("/usr/share/fonts/truetype/noto/NotoSansBengali-Bold.ttf", 0),
        ("/System/Library/Fonts/Supplemental/Bangla Sangam MN.ttc", 0),
    ]),
    ((0x0A80, 0x0AFF), [  # Gujarati
        ("/usr/share/fonts/truetype/noto/NotoSansGujarati-Bold.ttf", 0),
        ("/System/Library/Fonts/Supplemental/Gujarati Sangam MN.ttc", 0),
    ]),
    ((0x0B80, 0x0BFF), [  # Tamil
        ("/usr/share/fonts/truetype/noto/NotoSansTamil-Bold.ttf", 0),
        ("/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc", 0),
    ]),
    ((0x0C00, 0x0C7F), [  # Telugu
        ("/usr/share/fonts/truetype/noto/NotoSansTelugu-Bold.ttf", 0),
        ("/System/Library/Fonts/Supplemental/Telugu Sangam MN.ttc", 0),
    ]),
    ((0x0C80, 0x0CFF), [  # Kannada
        ("/usr/share/fonts/truetype/noto/NotoSansKannada-Bold.ttf", 0),
        ("/System/Library/Fonts/Supplemental/Kannada Sangam MN.ttc", 0),
    ]),
    ((0x0D00, 0x0D7F), [  # Malayalam
        ("/usr/share/fonts/truetype/noto/NotoSansMalayalam-Bold.ttf", 0),
        ("/System/Library/Fonts/Supplemental/Malayalam Sangam MN.ttc", 0),
    ]),
    ((0x0600, 0x06FF), [  # Arabic (also covers Urdu/Persian text)
        ("/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf", 0),
        ("/System/Library/Fonts/GeezaPro.ttc", 1),
    ]),
    ((0x4E00, 0x9FFF), [  # CJK (Chinese/Japanese) — not installed in the Docker
        # image (fonts-noto-cjk is ~150MB+, too heavy for the free-tier build);
        # falls through to the Latin default below, which will render as tofu.
        # Install fonts-noto-cjk and add candidates here if CJK on-screen text
        # is needed.
    ]),
]


def _detect_script_candidates(text: str) -> list:
    for (lo, hi), candidates in SCRIPT_FONT_CANDIDATES:
        if any(lo <= ord(ch) <= hi for ch in text):
            return candidates
    return []


def _load_font(size: int, text: str = "") -> ImageFont.FreeTypeFont:
    layout_engine = ImageFont.Layout.RAQM if _RAQM_AVAILABLE else ImageFont.Layout.BASIC
    for path, index in _detect_script_candidates(text):
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=index, layout_engine=layout_engine)
    for path, index in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=index, layout_engine=layout_engine)
    return ImageFont.load_default()


def render_text_card(text: str, width: int, height: int, out_path: Path) -> Path:
    """Renders `text` inside a semi-transparent bar near the bottom third of a
    `width`x`height` transparent canvas, sized for the video's own resolution."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_size = max(28, width // 18)
    font = _load_font(font_size, text)

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


def _wrapped_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    max_chars = max(5, int(len(text) * max_width / max(1, draw.textbbox((0, 0), text, font=font)[2])))
    return textwrap.wrap(text, width=max_chars) or [text]


def _draw_centered_lines(draw, lines, font, fill, top_y, width, line_gap) -> int:
    """Draws `lines` centered horizontally starting at `top_y`; returns the y
    position just below the last line."""
    y = top_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((width - text_w) // 2, y), line, font=font, fill=fill)
        y += text_h + line_gap
    return y


def render_hook_card(
    title: str,
    stat_text: str,
    width: int,
    height: int,
    out_path: Path,
    background_path: Path = None,
) -> Path:
    """Renders a bold, full-bleed branded "hook card": used as both the opening
    ~2s clip of a video (to grab attention before the narration starts) and as
    the standalone downloadable YouTube thumbnail for the same video.

    `background_path` (optional): a still image to crop/cover behind the text,
    e.g. the video's own first-scene visual, darkened for text contrast.
    """
    if background_path and Path(background_path).exists():
        bg = Image.open(background_path).convert("RGB")
        src_w, src_h = bg.size
        scale = max(width / src_w, height / src_h)
        bg = bg.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))))
        left = (bg.width - width) // 2
        top = (bg.height - height) // 2
        bg = bg.crop((left, top, left + width, top + height))
        img = bg.convert("RGBA")
        dark = Image.new("RGBA", (width, height), (10, 15, 25, 150))
        img = Image.alpha_composite(img, dark)
    else:
        img = Image.new("RGBA", (width, height), (12, 18, 30, 255))

    draw = ImageDraw.Draw(img)

    # Bold white title bar near the top, like a headline.
    title_font = _load_font(max(36, width // 14), title)
    title_lines = _wrapped_lines(draw, title.upper(), title_font, int(width * 0.88))
    title_bar_y0 = int(height * 0.08)
    line_h = draw.textbbox((0, 0), "A", font=title_font)[3]
    title_bar_h = len(title_lines) * (line_h + 10) + int(width * 0.05)
    draw.rectangle([0, title_bar_y0, width, title_bar_y0 + title_bar_h], fill=(255, 255, 255, 235))
    _draw_centered_lines(
        draw, title_lines, title_font, (15, 20, 35, 255),
        title_bar_y0 + int(width * 0.025), width, 10,
    )

    # Bright green stat callout near the bottom, for the "FY27 TARGET: ..."-style hook.
    if stat_text:
        stat_font = _load_font(max(30, width // 20), stat_text)
        stat_lines = _wrapped_lines(draw, stat_text.upper(), stat_font, int(width * 0.9))
        stat_line_h = draw.textbbox((0, 0), "A", font=stat_font)[3]
        stat_block_h = len(stat_lines) * (stat_line_h + 8)
        stat_y0 = height - stat_block_h - int(height * 0.08)
        draw.rectangle([0, stat_y0 - int(height * 0.02), width, height], fill=(5, 10, 18, 210))
        _draw_centered_lines(draw, stat_lines, stat_font, (60, 230, 110, 255), stat_y0, width, 8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)
    return out_path
