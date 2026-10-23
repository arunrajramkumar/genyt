"""Renders short on-screen text (facts/figures) as a transparent PNG overlay
using Pillow, since this ffmpeg build has no libass/freetype for drawtext.
"""
import re
import textwrap
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

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
        if ch in _ZERO_WIDTH_CHARS or ch == "️":
            continue
        # Emoji/pictographs (category So) and unassigned/private-use codepoints
        # aren't in the render fonts either and draw as a tofu box.
        if unicodedata.category(ch) in ("So", "Cn", "Co", "Cs"):
            continue
        out.append(" " if (ch != " " and ch.isspace()) else ch)
    return re.sub(r" {2,}", " ", "".join(out)).strip()


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
    """Greedy word wrap measured in real pixels, so no line exceeds max_width
    (unless a single word alone is wider)."""
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [text]


def _fit_title(draw, title: str, start_size: int, min_size: int, max_width: int, max_lines: int):
    """Largest font (shrinking from start_size) whose wrapped title fits
    max_width on every line within max_lines. Returns (font, lines)."""
    size = start_size
    while True:
        font = _load_font(size, title)
        lines = _wrapped_lines(draw, title, font, max_width)
        fits = all(draw.textbbox((0, 0), ln, font=font)[2] <= max_width for ln in lines)
        if (fits and len(lines) <= max_lines) or size <= min_size:
            return font, lines
        size = max(min_size, size - 4)


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


# Hook card accent palette -- red badge + angled banner (the better-tested
# high-CTR "alert" structure from the first mockup round), blended with a
# glowing-green stat number and frosted-glass title bar after a real
# AI-generated reference thumbnail (cooler "premium fintech" look, no flat
# opaque bars) showed that look reading as more professional for a finance
# stat callout than the original flat-yellow starburst.
_HOOK_BADGE_COLOR = (235, 55, 60, 255)
_HOOK_STAT_GLOW_COLOR = (80, 255, 150, 255)
_HOOK_STRIPE_COLOR = (50, 220, 120, 255)
_HOOK_BANNER_SKEW_FRACTION = 0.06

# "calm" style palette -- gold-on-dark-green, no angled badge/banner, for
# SIP/wealth-building/personal-finance content where the "breaking news"
# red-alert urgency of the default style reads as tonally wrong (modeled on
# a real AI-generated reference thumbnail for that content type: stacked
# gold-bevelled headline bars, no slanted badge, warm instead of stark vignette).
_CALM_ACCENT_COLOR = (230, 180, 70, 255)
_CALM_PANEL_COLOR = (8, 16, 10, 215)
_CALM_TINT_COLOR = (20, 28, 15, 90)

_STAT_NUMBER_RE = re.compile(r"[₹$]?[\d][\d,]*\.?\d*%?")


def _draw_frosted_bar(img: Image.Image, y0: int, y1: int, width: int) -> None:
    """Composites a translucent light-to-blue gradient band with a bright
    top edge -- a "frosted glass" panel the background photo shows faintly
    through, instead of an opaque flat-white bar that hides it completely."""
    bar_h = y1 - y0
    bar = Image.new("RGBA", (1, bar_h))
    for y in range(bar_h):
        t = y / max(1, bar_h)
        bar.putpixel((0, y), (
            int(245 - 50 * t), int(250 - 35 * t), int(255 - 20 * t), int(215 - 35 * t),
        ))
    bar = bar.resize((width, bar_h))
    img.alpha_composite(bar, (0, y0))
    glass_draw = ImageDraw.Draw(img)
    glass_draw.line([(0, y0), (width, y0)], fill=(255, 255, 255, 170), width=2)


def _draw_glow_text(img: Image.Image, xy, text, font, fill, blur_radius=10) -> None:
    """Draws `text` with a soft colored glow halo behind a crisp fill --
    the "neon stat number" look, vs. a hard outline."""
    x, y = xy
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).text((x, y), text, font=font, fill=fill)
    glow = glow.filter(ImageFilter.GaussianBlur(blur_radius))
    img.alpha_composite(glow)
    ImageDraw.Draw(img).text((x, y), text, font=font, fill=fill)


def _apply_vignette(img: Image.Image, width: int, height: int) -> Image.Image:
    """Darkens the top and bottom bands (where title/stat text sit) while
    keeping the middle band comparatively bright, so a background photo still
    reads as a photo instead of disappearing under one flat overlay."""
    vignette = Image.new("L", (1, height))
    for y in range(height):
        t = y / height
        if t < 0.32:
            alpha = int(170 * (1 - t / 0.32))
        elif t > 0.62:
            alpha = int(190 * ((t - 0.62) / 0.38))
        else:
            alpha = 40
        vignette.putpixel((0, y), alpha)
    vignette = vignette.resize((width, height))
    dark = Image.new("RGBA", (width, height), (5, 8, 15, 255))
    dark.putalpha(vignette)
    return Image.alpha_composite(img, dark)


def _crop_to_cover(background_path: Path, width: int, height: int) -> Image.Image:
    bg = Image.open(background_path).convert("RGB")
    src_w, src_h = bg.size
    scale = max(width / src_w, height / src_h)
    bg = bg.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))))
    left = (bg.width - width) // 2
    top = (bg.height - height) // 2
    return bg.crop((left, top, left + width, top + height)).convert("RGBA")


def _render_hook_card_calm(
    title: str, stat_text: str, width: int, height: int, out_path: Path,
    background_path: Path = None, badge_text: str = None,
) -> Path:
    if background_path and Path(background_path).exists():
        img = _crop_to_cover(background_path, width, height)
    else:
        img = Image.new("RGBA", (width, height), (10, 20, 14, 255))
    tint = Image.new("RGBA", (width, height), _CALM_TINT_COLOR)
    img = Image.alpha_composite(img, tint)
    draw = ImageDraw.Draw(img)

    pad_x, pad_y = int(width * 0.04), int(width * 0.02)
    title_font, title_lines = _fit_title(
        draw, title.upper(), max(40, width // 13), max(28, width // 24),
        int(width * 0.92) - pad_x * 2, max_lines=3,
    )
    line_h = draw.textbbox((0, 0), "HgÅ", font=title_font)[3]
    y = int(height * 0.08)
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        text_w = bbox[2] - bbox[0]
        bar_x0 = (width - (text_w + pad_x * 2)) // 2
        bar_h = line_h + pad_y * 2
        draw.rounded_rectangle(
            [bar_x0, y, bar_x0 + text_w + pad_x * 2, y + bar_h],
            radius=16, fill=_CALM_PANEL_COLOR, outline=_CALM_ACCENT_COLOR, width=4,
        )
        draw.text(((width - text_w) // 2 - bbox[0], y + pad_y), line, font=title_font, fill=_CALM_ACCENT_COLOR)
        y += bar_h + 14

    if badge_text:
        badge_font = _load_font(max(24, width // 30), badge_text)
        bbox = draw.textbbox((0, 0), badge_text.upper(), font=badge_font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = int(width * 0.025), int(width * 0.012)
        bx0 = int(width * 0.04)
        draw.rounded_rectangle(
            [bx0, y + 10, bx0 + text_w + pad_x * 2, y + 10 + text_h + pad_y * 2],
            radius=10, fill=(15, 20, 12, 220),
        )
        draw.text((bx0 + pad_x, y + 10 + pad_y), badge_text.upper(), font=badge_font, fill=_CALM_ACCENT_COLOR)

    if stat_text:
        panel_y0, panel_y1 = int(height * 0.76), height - int(height * 0.04)
        draw.rounded_rectangle(
            [int(width * 0.06), panel_y0, int(width * 0.94), panel_y1],
            radius=28, fill=_CALM_PANEL_COLOR, outline=_CALM_ACCENT_COLOR, width=3,
        )
        num_font = _load_font(max(60, width // 9), stat_text)
        bbox = draw.textbbox((0, 0), stat_text, font=num_font)
        num_w, num_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cy = (panel_y0 + panel_y1) // 2 - num_h // 2
        _draw_glow_text(img, ((width - num_w) // 2, cy), stat_text, num_font, _CALM_ACCENT_COLOR, blur_radius=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)
    return out_path


def render_hook_card(
    title: str,
    stat_text: str,
    width: int,
    height: int,
    out_path: Path,
    background_path: Path = None,
    badge_text: str = None,
    style: str = "alert",
) -> Path:
    """Renders a bold, full-bleed branded "hook card": used as both the opening
    ~2s clip of a video (to grab attention before the narration starts) and as
    the standalone downloadable YouTube thumbnail for the same video.

    `background_path` (optional): a still image to crop/cover behind the text,
    e.g. the video's own first-scene visual, with a vignette for text contrast.
    `badge_text` (optional): a short label stamped in a corner pill, e.g.
    "CBSE | GRADE 6 MATHS" for an education-channel video, so series/syllabus
    identity is visible even in the small thumbnail grid on a channel page.
    `style`: "alert" (default) is the red-badge/angled-banner/glow-green-stat
    "breaking news" treatment; "calm" is a gold-on-dark-green treatment with
    no badge/banner angling, for SIP/wealth-building/personal-finance content
    where the alert urgency reads as tonally wrong.
    """
    title = sanitize_display_text(title)
    stat_text = sanitize_display_text(stat_text) if stat_text else stat_text
    badge_text = sanitize_display_text(badge_text) if badge_text else badge_text
    if style == "calm":
        return _render_hook_card_calm(
            title, stat_text, width, height, out_path,
            background_path=background_path, badge_text=badge_text,
        )
    if background_path and Path(background_path).exists():
        img = _crop_to_cover(background_path, width, height)
    else:
        img = Image.new("RGBA", (width, height), (12, 18, 30, 255))
    img = _apply_vignette(img, width, height)
    draw = ImageDraw.Draw(img)

    # Badge: an angled pill (parallelogram) instead of a flat rectangle, for a
    # more dynamic look -- geometry is needed before the title (below) so the
    # title can start clear of it, same as before this redesign.
    badge_bottom = 0
    if badge_text:
        badge_font = _load_font(max(26, width // 26), badge_text)
        text_bbox = draw.textbbox((0, 0), badge_text.upper(), font=badge_font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        pad_x, pad_y = int(width * 0.03), int(width * 0.018)
        badge_x0, badge_y0 = int(width * 0.04), int(width * 0.04)
        badge_w, badge_h = text_w + pad_x * 2, text_h + pad_y * 2
        skew = int(badge_h * 0.35)
        draw.polygon(
            [
                (badge_x0 + skew, badge_y0), (badge_x0 + badge_w, badge_y0),
                (badge_x0 + badge_w - skew, badge_y0 + badge_h), (badge_x0, badge_y0 + badge_h),
            ],
            fill=_HOOK_BADGE_COLOR, outline=(0, 0, 0, 255), width=4,
        )
        draw.text((badge_x0 + pad_x + skew // 2, badge_y0 + pad_y), badge_text.upper(), font=badge_font, fill=(255, 255, 255, 255))
        badge_bottom = badge_y0 + badge_h

    # Title: bold text on a translucent frosted-glass bar -- keeps the photo
    # faintly visible behind it (unlike the old opaque white bar) while still
    # giving the text a clean, high-contrast panel to sit on.
    title_font = _load_font(max(36, width // 13), title)
    title_lines = _wrapped_lines(draw, title.upper(), title_font, int(width * 0.88))
    line_h = draw.textbbox((0, 0), "A", font=title_font)[3]
    bar_h = len(title_lines) * (line_h + 10) + int(width * 0.05)
    bar_y0 = max(int(height * 0.08), badge_bottom + int(height * 0.02))
    _draw_frosted_bar(img, bar_y0, bar_y0 + bar_h, width)
    draw = ImageDraw.Draw(img)
    _draw_centered_lines(draw, title_lines, title_font, (15, 20, 35, 255), bar_y0 + int(width * 0.025), width, 10)

    # Stat: a glowing green number (no flat starburst) is the hero element,
    # with any trailing label (e.g. "Revenue") in a small pill above it --
    # modeled on a real AI-generated finance thumbnail's "FY27 TARGET: ₹600
    # CR" treatment, which reads as more premium than a flat colored block.
    if stat_text:
        match = _STAT_NUMBER_RE.search(stat_text)
        if match:
            number = match.group(0)
            label = (stat_text[:match.start()] + stat_text[match.end():]).strip()
        else:
            number, label = stat_text, ""

        banner_y0 = int(height * 0.74)
        banner_y1 = height - int(height * 0.03)
        skew = int(width * _HOOK_BANNER_SKEW_FRACTION)
        draw.polygon(
            [(0, banner_y0 + skew), (width, banner_y0), (width, banner_y1), (0, banner_y1 - skew)],
            fill=(10, 14, 20, 225),
        )
        draw.polygon(
            [(0, banner_y0 + skew), (width, banner_y0), (width, banner_y0 + 10), (0, banner_y0 + skew + 10)],
            fill=_HOOK_STRIPE_COLOR,
        )

        content_top = banner_y0 + int(height * 0.035)
        if label:
            label_font = _load_font(max(26, width // 24), label)
            label_lines = _wrapped_lines(draw, label.upper(), label_font, int(width * 0.8))
            label_bbox = draw.textbbox((0, 0), label_lines[0], font=label_font)
            label_w, label_h = label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1]
            pad_x, pad_y = int(width * 0.025), int(width * 0.012)
            pill_x0 = (width - label_w) // 2 - pad_x
            pill_x1 = (width + label_w) // 2 + pad_x
            draw.rounded_rectangle(
                [pill_x0, content_top, pill_x1, content_top + label_h + pad_y * 2],
                radius=10, fill=(235, 240, 248, 225),
            )
            draw.text(((width - label_w) // 2, content_top + pad_y), label_lines[0].upper(), font=label_font, fill=(15, 20, 35, 255))
            content_top += label_h + pad_y * 2 + int(height * 0.015)

        num_font = _load_font(max(64, width // 8), number)
        num_bbox = draw.textbbox((0, 0), number, font=num_font)
        num_w, num_h = num_bbox[2] - num_bbox[0], num_bbox[3] - num_bbox[1]
        _draw_glow_text(img, ((width - num_w) // 2, content_top), number, num_font, _HOOK_STAT_GLOW_COLOR)
        draw = ImageDraw.Draw(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path)
    return out_path


_THUMB_UP_COLOR = (70, 240, 140, 255)
_THUMB_DOWN_COLOR = (255, 85, 85, 255)
_THUMB_NEUTRAL_COLOR = (255, 215, 64, 255)
_THUMB_HOOK_COLOR = (255, 215, 64, 255)
_THUMB_TAG_COLORS = {"IPO": (130, 70, 230, 255)}
_THUMB_TAG_DEFAULT_COLOR = (230, 45, 55, 255)


def _fit_font(draw, text: str, max_width: int, start_size: int, min_size: int):
    size = start_size
    while size > min_size:
        font = _load_font(size, text)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=max(2, size // 18))
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 4
    return _load_font(min_size, text)


def _fit_lines(draw, text: str, max_width: int, start_size: int, min_size: int, max_lines: int):
    """Biggest font at which `text` wraps into at most `max_lines` lines."""
    size = start_size
    while size >= min_size:
        font = _load_font(size, text)
        lines = _wrapped_lines(draw, text, font, max_width)
        if len(lines) <= max_lines:
            return font, lines
        size -= 4
    font = _load_font(min_size, text)
    return font, _wrapped_lines(draw, text, font, max_width)[:max_lines]


def _draw_stroked_centered(draw, text, font, fill, y, width, stroke=None):
    stroke = stroke if stroke is not None else max(2, font.size // 18)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    draw.text(((width - (bbox[2] - bbox[0])) // 2 - bbox[0], y - bbox[1]), text, font=font, fill=fill,
              stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    return bbox[3] - bbox[1]


def _darken_for_text(img: Image.Image, width: int, height: int) -> Image.Image:
    """A uniform dim plus a heavier fade toward the bottom, so white/green text
    stays readable over any AI or stock background."""
    shade = Image.new("L", (1, height))
    for y in range(height):
        t = y / height
        shade.putpixel((0, y), int(95 + 120 * max(0.0, (t - 0.45) / 0.55)))
    shade = shade.resize((width, height))
    dark = Image.new("RGBA", (width, height), (4, 6, 12, 255))
    dark.putalpha(shade)
    return Image.alpha_composite(img, dark)


def _scrim(width: int, height: int, y0: int, y1: int, alpha: int = 130) -> Image.Image:
    """Transparent canvas with a soft dark band from y0 to y1 (feathered
    edges), so overlay text stays readable over any moving footage."""
    band = Image.new("L", (1, height), 0)
    feather = max(1, int(height * 0.06))
    for y in range(height):
        if y0 <= y <= y1:
            a = alpha
        elif y0 - feather < y < y0:
            a = int(alpha * (y - (y0 - feather)) / feather)
        elif y1 < y < y1 + feather:
            a = int(alpha * ((y1 + feather) - y) / feather)
        else:
            a = 0
        band.putpixel((0, y), a)
    img = Image.new("RGBA", (width, height), (4, 6, 12, 255))
    img.putalpha(band.resize((width, height)))
    return img


def render_stock_thumbnail(spec: dict, width: int, height: int, out_path: Path, background_path: Path = None,
                           transparent: bool = False) -> Path:
    """Stock-video thumbnail/hook card from a pipeline.thumbnail_spec spec:
    results/IPO tag, the company's name, one giant colour-coded YoY number
    with an arrow and a plain label, two metric chips and a short question --
    the elements a viewer scanning search results can take in at a glance."""
    if transparent:
        # In-video opener: text over the moving first scene, not a slide.
        img = _scrim(width, height, int(height * 0.03), int(height * 0.62))
    elif background_path and Path(background_path).exists():
        img = _darken_for_text(_crop_to_cover(background_path, width, height), width, height)
    else:
        img = _darken_for_text(Image.new("RGBA", (width, height), (12, 18, 30, 255)), width, height)
    draw = ImageDraw.Draw(img)
    content_w = int(width * 0.86)

    y = int(height * 0.05)
    tag = sanitize_display_text(spec.get("tag") or "")
    if tag:
        tag_font = _load_font(max(28, width // 20), tag)
        bbox = draw.textbbox((0, 0), tag, font=tag_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = int(width * 0.035), int(width * 0.02)
        x0 = (width - tw) // 2 - pad_x
        draw.rounded_rectangle([x0, y, x0 + tw + pad_x * 2, y + th + pad_y * 2], radius=14,
                               fill=_THUMB_TAG_COLORS.get(tag, _THUMB_TAG_DEFAULT_COLOR))
        draw.text((x0 + pad_x - bbox[0], y + pad_y - bbox[1]), tag, font=tag_font, fill=(255, 255, 255, 255))
        y += th + pad_y * 2 + int(height * 0.025)

    name = sanitize_display_text(spec.get("name") or "")
    if name:
        name_font, name_lines = _fit_lines(draw, name, content_w, max(60, width // 9), max(40, width // 16), 2)
        for line in name_lines:
            y += _draw_stroked_centered(draw, line, name_font, (255, 255, 255, 255), y, width) + int(height * 0.012)

    direction = spec.get("direction")
    color = {"up": _THUMB_UP_COLOR, "down": _THUMB_DOWN_COLOR}.get(direction, _THUMB_NEUTRAL_COLOR)
    value = sanitize_display_text(spec.get("hero_value") or "")
    hero_top = max(y + int(height * 0.06), int(height * 0.36))
    arrow_w = int(width * 0.16) if direction else 0
    gap = int(width * 0.03) if direction else 0
    value_font = _fit_font(draw, value, content_w - arrow_w - gap, max(120, width // 3), max(80, width // 7))
    vb = draw.textbbox((0, 0), value, font=value_font)
    vw, vh = vb[2] - vb[0], vb[3] - vb[1]
    x0 = (width - (arrow_w + gap + vw)) // 2
    if direction:
        ah = int(vh * 0.8)
        ax0, ay0 = x0, hero_top + (vh - ah) // 2
        points = ([(ax0, ay0 + ah), (ax0 + arrow_w, ay0 + ah), (ax0 + arrow_w // 2, ay0)] if direction == "up"
                  else [(ax0, ay0), (ax0 + arrow_w, ay0), (ax0 + arrow_w // 2, ay0 + ah)])
        draw.polygon(points, fill=color, outline=(0, 0, 0, 255), width=4)
    _draw_glow_text(img, (x0 + arrow_w + gap - vb[0], hero_top - vb[1]), value, value_font, color, blur_radius=14)
    draw = ImageDraw.Draw(img)
    y = hero_top + vh + int(height * 0.025)

    label = sanitize_display_text(spec.get("hero_label") or "")
    if label:
        label_font = _load_font(max(32, width // 17), label)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = int(width * 0.04), int(width * 0.018)
        lx0 = (width - lw) // 2 - pad_x
        draw.rounded_rectangle([lx0, y, lx0 + lw + pad_x * 2, y + lh + pad_y * 2], radius=16, fill=(245, 247, 250, 240))
        draw.text((lx0 + pad_x - bbox[0], y + pad_y - bbox[1]), label, font=label_font, fill=(12, 16, 28, 255))
        y += lh + pad_y * 2 + int(height * 0.035)

    chips = [sanitize_display_text(c) for c in spec.get("chips") or [] if c]
    if chips:
        chip_font = _load_font(max(30, width // 19), " ".join(chips))
        pad_x, pad_y, gap = int(width * 0.035), int(width * 0.018), int(width * 0.03)
        boxes = [draw.textbbox((0, 0), c, font=chip_font) for c in chips]
        widths = [b[2] - b[0] + pad_x * 2 for b in boxes]
        ch = max(b[3] - b[1] for b in boxes) + pad_y * 2
        x = (width - (sum(widths) + gap * (len(chips) - 1))) // 2
        for chip, bbox, w in zip(chips, boxes, widths):
            draw.rounded_rectangle([x, y, x + w, y + ch], radius=14, fill=(10, 14, 24, 215),
                                   outline=(255, 255, 255, 140), width=2)
            draw.text((x + pad_x - bbox[0], y + pad_y - bbox[1]), chip, font=chip_font, fill=(255, 255, 255, 255))
            x += w + gap
        y += ch + int(height * 0.035)

    hook = sanitize_display_text(spec.get("hook") or "")
    if hook:
        hook_font, hook_lines = _fit_lines(draw, hook, content_w, max(56, width // 12), max(36, width // 20), 2)
        hy = max(y, int(height * 0.72))
        for line in hook_lines:
            hy += _draw_stroked_centered(draw, line, hook_font, _THUMB_HOOK_COLOR, hy, width) + int(height * 0.01)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    (img if transparent else img.convert("RGB")).save(out_path)
    return out_path
