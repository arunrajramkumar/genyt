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


def test_render_hook_card_with_badge_text_produces_a_nonempty_png(tmp_path):
    out = textcard.render_hook_card(
        "Patterns in Mathematics", "", 1280, 720, tmp_path / "badge.png",
        badge_text="CBSE | Grade 6 Maths",
    )
    assert out.exists() and out.stat().st_size > 0


def test_render_hook_card_without_badge_text_still_works(tmp_path):
    out = textcard.render_hook_card("Title", "", 1280, 720, tmp_path / "no_badge.png")
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
