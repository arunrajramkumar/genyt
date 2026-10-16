"""Renders short on-screen text (facts/figures) as a transparent PNG overlay
using Pillow, since this ffmpeg build has no libass/freetype for drawtext.
"""
import re
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


# Characters that carry no visual width/shape at all -- Pillow still needs a
# cmap entry to draw "nothing", and most fonts don't define one, so these show
# up as a tofu/box glyph exactly like a missing letter would.
_ZERO_WIDTH_CHARS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\ufeff",  # zero width no-break space / BOM
    "\u00ad",  # soft hyphen
}


def sanitize_display_text(text: str) -> str:
    """Pillow draws a visible tofu/box glyph for any character missing from
    the active font's cmap. LLM output occasionally uses exotic Unicode
    whitespace for "Indian English" number formatting -- narrow no-break
    space (U+202F, e.g. "Sep 18", "88 crore"), non-breaking space (U+00A0),
    thin space (U+2009), etc. -- which common render fonts (Noto Sans Bold,
    Helvetica) don't map, so what should look like a plain space shows up as
    a stray box on screen (reported: a box between "Price" and "92" in a
    published video). Normalize every Unicode whitespace character to a
    plain ASCII space, and drop zero-width/format characters outright since
    they have no visual meaning to begin with."""
    out = []
    for ch in text:
        if ch in _ZERO_WIDTH_CHARS:
            continue
        out.append(" " if (ch != " " and ch.isspace()) else ch)
    return "".join(out)


def _load_font(size: int, text: str = "") -> ImageFont.FreeTypeFont:
    layout_engine = ImageFont.Layout.RAQM if _RAQM_AVAILABLE else ImageFont.Layout.BASIC
    for path, index in _detect_script_candidates(text):
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=index, layout_engine=layout_engine)
    for path, index in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=index, layout_engine=layout_engine)
    return ImageFont.load_default()


# Same bright green used for the hook card's stat callout, so a number
# standing out mid-sentence in an on-screen text card ("Revenue grew to ₹412
# Cr") reads as the same brand "this is the number that matters" cue the
# audience already sees in the hook card and on the thumbnail.
NUMBER_ACCENT_COLOR = (60, 230, 110, 255)
_NUMBER_TOKEN_RE = re.compile(r"[\d₹%]")


def _draw_centered_line_with_number_accent(draw, line: str, font, default_fill, y: int, width: int) -> None:
    """Draws `line` centered horizontally at `y`, coloring any token that
    contains a digit/₹/% in NUMBER_ACCENT_COLOR instead of `default_fill`."""
    tokens = line.split(" ")
    widths = [draw.textbbox((0, 0), tok, font=font)[2] for tok in tokens]
    space_w = draw.textbbox((0, 0), " ", font=font)[2]
    total_w = sum(widths) + space_w * (len(tokens) - 1)
    x = (width - total_w) // 2
    for tok, tok_w in zip(tokens, widths):
        fill = NUMBER_ACCENT_COLOR if _NUMBER_TOKEN_RE.search(tok) else default_fill
        draw.text((x, y), tok, font=font, fill=fill)
        x += tok_w + space_w


def render_text_card(text: str, width: int, height: int, out_path: Path) -> Path:
    """Renders `text` inside a semi-transparent bar near the bottom third of a
    `width`x`height` transparent canvas, sized for the video's own resolution."""
    text = sanitize_display_text(text)
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
        _draw_centered_line_with_number_accent(draw, line, font, (255, 255, 255, 255), y, width)
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


def render_watermark(text: str, width: int, height: int, out_path: Path) -> Path:
    """Renders a small, semi-transparent channel-handle pill for the top-right
    corner, present for the whole video rather than just the hook/thumbnail --
    a re-shared or screen-recorded clip (common for finance Shorts) otherwise
    carries no indication of which channel it came from."""
    text = sanitize_display_text(text)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_size = max(18, width // 32)
    font = _load_font(font_size, text)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad_x, pad_y = int(font_size * 0.7), int(font_size * 0.4)
    margin = int(width * 0.03)
    x1 = width - margin
    y0 = margin
    x0 = x1 - text_w - pad_x * 2
    y1 = y0 + text_h + pad_y * 2

    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 130), outline=None)
    draw.text((x0 + pad_x, y0 + pad_y), text, font=font, fill=(255, 255, 255, 220))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def render_hook_card(
    title: str,
    stat_text: str,
    width: int,
    height: int,
    out_path: Path,
    background_path: Path = None,
    badge_text: str = None,
) -> Path:
    """Renders a bold, full-bleed branded "hook card": used as both the opening
    ~2s clip of a video (to grab attention before the narration starts) and as
    the standalone downloadable YouTube thumbnail for the same video.

    `background_path` (optional): a still image to crop/cover behind the text,
    e.g. the video's own first-scene visual, darkened for text contrast.
    `badge_text` (optional): a short label stamped in a corner pill, e.g.
    "CBSE | GRADE 6 MATHS" for an education-channel video, so series/syllabus
    identity is visible even in the small thumbnail grid on a channel page.
    """
    title = sanitize_display_text(title)
    stat_text = sanitize_display_text(stat_text) if stat_text else stat_text
    badge_text = sanitize_display_text(badge_text) if badge_text else badge_text
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

    # Badge geometry is needed before laying out the title bar (below) so the
    # title can start clear of it -- previously the title bar always started
    # at a fixed 0.08*height regardless of the badge, so a badge tall enough
    # (e.g. a two-word syllabus label) visually overlapped the first line of
    # the title text drawn on top of it.
    badge_bottom = 0
    if badge_text:
        badge_font = _load_font(max(22, width // 28), badge_text)
        text_bbox = draw.textbbox((0, 0), badge_text.upper(), font=badge_font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        pad_x, pad_y = int(width * 0.025), int(width * 0.015)
        badge_x0, badge_y0 = int(width * 0.03), int(width * 0.03)
        badge_bottom = badge_y0 + text_h + pad_y * 2

    # Bold white title bar near the top, like a headline.
    title_font = _load_font(max(36, width // 14), title)
    title_lines = _wrapped_lines(draw, title.upper(), title_font, int(width * 0.88))
    title_bar_y0 = max(int(height * 0.08), badge_bottom + int(height * 0.02))
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

    if badge_text:
        draw.rectangle(
            [badge_x0, badge_y0, badge_x0 + text_w + pad_x * 2, badge_y0 + text_h + pad_y * 2],
            fill=(230, 60, 50, 255),
        )
        draw.text((badge_x0 + pad_x, badge_y0 + pad_y), badge_text.upper(), font=badge_font, fill=(255, 255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)
    return out_path
