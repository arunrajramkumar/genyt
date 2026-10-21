from pathlib import Path

from pipeline import textcard


def test_detect_script_candidates_latin_falls_through():
    # No script range matches plain Latin text — caller falls back to FONT_CANDIDATES.
    assert textcard._detect_script_candidates("Revenue: Rs 412 Cr") == []


def test_detect_script_candidates_devanagari():
    candidates = textcard._detect_script_candidates("गोपाल की कहानी")
    assert candidates, "Devanagari text must match a script font list"
    assert candidates == textcard.SCRIPT_FONT_CANDIDATES[0][1]


def test_detect_script_candidates_tamil():
    candidates = textcard._detect_script_candidates("ஒரு குறுகிய கதை")
    tamil_entry = next(c for (lo, hi), c in textcard.SCRIPT_FONT_CANDIDATES if lo == 0x0B80)
    assert candidates == tamil_entry


def test_detect_script_candidates_arabic():
    candidates = textcard._detect_script_candidates("مرحبا بالعالم")
    arabic_entry = next(c for (lo, hi), c in textcard.SCRIPT_FONT_CANDIDATES if lo == 0x0600)
    assert candidates == arabic_entry


def test_detect_script_candidates_mixed_text_matches_first_non_latin_script():
    # A Hindi sentence with an embedded ASCII number/punctuation should still
    # resolve to the Devanagari font list (digits/punctuation don't match any
    # script range and are skipped).
    candidates = textcard._detect_script_candidates("राजस्व: Rs 412 करोड़")
    assert candidates == textcard.SCRIPT_FONT_CANDIDATES[0][1]


def test_load_font_returns_a_usable_font_for_every_registered_script():
    samples = {
        "Latin": "Revenue: Rs 412 Cr",
        "Devanagari": "गोपाल की कहानी",
        "Bengali": "একটি ছোট গল্প",
        "Gujarati": "એક ટૂંકી વાર્તા",
        "Tamil": "ஒரு குறுகிய கதை",
        "Telugu": "ఒక చిన్న కథ",
        "Kannada": "ಒಂದು ಸಣ್ಣ ಕಥೆ",
        "Malayalam": "ഒരു ചെറിയ കഥ",
        "Arabic": "قصة قصيرة",
    }
    for label, text in samples.items():
        font = textcard._load_font(40, text)
        assert font is not None, f"{label} text must resolve to a loadable font"


def test_render_text_card_produces_a_nonempty_png(tmp_path):
    out = textcard.render_text_card("Revenue: Rs 412 Cr", 720, 1280, tmp_path / "card.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_text_card_handles_non_latin_scripts_without_crashing(tmp_path):
    for i, text in enumerate([
        "राजस्व: ₹412 करोड़",
        "ஒரு குறுகிய கதை",
        "قصة قصيرة",
    ]):
        out = textcard.render_text_card(text, 720, 1280, tmp_path / f"card_{i}.png")
        assert out.exists() and out.stat().st_size > 0


def test_render_text_card_wraps_long_text_into_multiple_lines(tmp_path):
    long_text = "This is a very long piece of on-screen text that should wrap across multiple lines"
    out = textcard.render_text_card(long_text, 720, 1280, tmp_path / "wrap.png")
    assert out.exists()


def test_render_hook_card_with_and_without_background(tmp_path):
    out_no_bg = textcard.render_hook_card(
        "Steel Co: Order Book Surges", "₹2,400 crore order book",
        720, 1280, tmp_path / "hook_no_bg.png",
    )
    assert out_no_bg.exists() and out_no_bg.stat().st_size > 0

    out_empty_stat = textcard.render_hook_card(
        "Steel Co: Order Book Surges", "",
        720, 1280, tmp_path / "hook_no_stat.png",
    )
    assert out_empty_stat.exists()


def test_rupee_glyph_font_candidates_exist_on_at_least_one_platform():
    # Regression guard for the known Arial-Bold-tofu rupee glyph bug: at least
    # one of the configured Latin fallback fonts must actually be present.
    assert any(Path(p).exists() for p, _ in textcard.FONT_CANDIDATES)


# -- sanitize_display_text: strip the characters that render as a tofu box ----

def test_sanitize_display_text_normalizes_narrow_no_break_space():
    # Regression for a real published video: a narrow no-break space (used by
    # the model for "Indian English" number formatting, e.g. "Price 92") has
    # no glyph in Noto Sans/Helvetica and rendered as a visible box.
    assert textcard.sanitize_display_text("Price 92") == "Price 92"


def test_sanitize_display_text_normalizes_non_breaking_space():
    assert textcard.sanitize_display_text("Rs 412") == "Rs 412"


def test_sanitize_display_text_normalizes_thin_and_other_unicode_spaces():
    for space in (" ", " ", " ", " ", "　"):
        assert textcard.sanitize_display_text(f"A{space}B") == "A B"


def test_sanitize_display_text_drops_zero_width_characters():
    assert textcard.sanitize_display_text("Price​92") == "Price92"
    assert textcard.sanitize_display_text("A﻿B­C") == "ABC"


def test_sanitize_display_text_leaves_plain_text_untouched():
    assert textcard.sanitize_display_text("Revenue: Rs 412 Cr") == "Revenue: Rs 412 Cr"


def test_render_text_card_sanitizes_narrow_no_break_space_before_measuring(tmp_path, monkeypatch):
    # The sanitized text, not the raw text, must be what's wrapped/measured/
    # drawn -- patch render to capture what actually reaches draw.text.
    seen = {}
    from PIL import ImageDraw
    original_text = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **k):
        seen.setdefault("texts", []).append(text)
        return original_text(self, xy, text, *a, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    textcard.render_text_card("Price 92", 720, 1280, tmp_path / "card.png")
    assert any(" " not in t for t in seen["texts"])
    assert not any(" " in t for t in seen["texts"])


def test_render_hook_card_sanitizes_title_stat_and_badge_text(tmp_path):
    out = textcard.render_hook_card(
        "Price Target", "₹92 Crore", 720, 1280, tmp_path / "hook.png",
        badge_text="Q1​FY27",
    )
    assert out.exists() and out.stat().st_size > 0


def test_render_hook_card_with_badge_text_produces_a_nonempty_png(tmp_path):
    out = textcard.render_hook_card(
        "Patterns in Mathematics", "", 1280, 720, tmp_path / "badge.png",
        badge_text="CBSE | Grade 6 Maths",
    )
    assert out.exists() and out.stat().st_size > 0


def test_render_hook_card_without_badge_text_still_works(tmp_path):
    out = textcard.render_hook_card("Title", "", 1280, 720, tmp_path / "no_badge.png")
    assert out.exists() and out.stat().st_size > 0


def test_render_text_card_colors_number_tokens_with_accent(tmp_path, monkeypatch):
    # Numbers are the "this is the number that matters" cue -- they must be
    # drawn in NUMBER_ACCENT_COLOR, while surrounding plain words stay white.
    seen = []
    from PIL import ImageDraw
    original_text = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, fill=None, **k):
        seen.append((text, fill))
        return original_text(self, xy, text, *a, fill=fill, **k)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    textcard.render_text_card("Revenue grew to Rs412 Cr", 720, 1280, tmp_path / "card.png")

    fills_by_token = dict(seen)
    assert fills_by_token["Rs412"] == textcard.NUMBER_ACCENT_COLOR
    assert fills_by_token["Revenue"] == (255, 255, 255, 255)
    assert fills_by_token["Cr"] == (255, 255, 255, 255)


def test_render_watermark_produces_a_nonempty_png(tmp_path):
    out = textcard.render_watermark("@mychannel", 720, 1280, tmp_path / "wm.png")
    assert out.exists() and out.stat().st_size > 0


def test_render_hook_card_calm_style_produces_a_nonempty_png(tmp_path):
    out = textcard.render_hook_card(
        "Start a ₹5,000 SIP Today, Retire With ₹2 Crore", "₹2 Crore",
        1080, 1920, tmp_path / "calm.png", badge_text="Wealth Building",
        style="calm",
    )
    assert out.exists() and out.stat().st_size > 0


def test_render_hook_card_calm_style_works_without_stat_or_badge(tmp_path):
    out = textcard.render_hook_card(
        "The Power Of Compounding", "", 1080, 1920, tmp_path / "calm_plain.png",
        style="calm",
    )
    assert out.exists() and out.stat().st_size > 0


def test_render_hook_card_badge_does_not_overlap_title_bar(tmp_path):
    # Regression for a real generated video's opening frame: the red corner
    # badge sat at a fixed top-left position while the white title bar always
    # started at a fixed 0.08*height regardless of the badge, so a badge tall
    # enough overlapped/obscured the first line of the centered title text.
    # The title bar must now start below the badge's bottom edge whenever a
    # badge is present.
    from PIL import Image

    width, height = 1280, 720
    out = textcard.render_hook_card(
        "Discovering Patterns: From Fibonacci To Squares In Everyday Math", "",
        width, height, tmp_path / "badge_overlap.png",
        badge_text="CBSE | Grade 6 Maths",
    )
    img = Image.open(out).convert("RGB")
    badge_y0 = int(width * 0.03)
    # Sample a row through the middle of the badge's vertical band, but far to
    # the right where the badge itself doesn't reach -- this pixel must still
    # be the plain background color, not the white title bar, otherwise the
    # title bar's top edge has crept up into the badge's row.
    pixel = img.getpixel((width - 10, badge_y0 + 5))
    assert pixel != (255, 255, 255), "title bar overlaps the badge's vertical band"


def test_sanitize_display_text_drops_emoji_and_collapses_spaces():
    assert textcard.sanitize_display_text("Wealth \U0001F4C8 builder ⚠️") == "Wealth builder"
    assert textcard.sanitize_display_text("Rs ₹39") == "Rs ₹39"


def test_calm_hook_card_title_lines_fit_inside_the_frame(tmp_path):
    from PIL import ImageDraw, Image
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    title = "32.3% ROCE: Steelcast Ltd - Long-Term Wealth Builder?".upper()
    font, lines = textcard._fit_title(draw, title, 83, 45, 900, max_lines=3)
    assert all(draw.textbbox((0, 0), ln, font=font)[2] <= 900 for ln in lines)
    out = textcard.render_hook_card(title, "0.5% | Rs 39", 1080, 1920, tmp_path / "c.png", style="calm")
    assert out.exists()
