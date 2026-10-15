from PIL import Image, ImageDraw

from pipeline import diagrams


# -- validate_diagram_spec: grounding / schema checks ---------------------------

def test_validate_diagram_spec_rejects_missing_kind():
    ok, reason = diagrams.validate_diagram_spec({}, source_text="anything")
    assert ok is False
    assert "kind" in reason


def test_validate_diagram_spec_rejects_unknown_kind():
    ok, reason = diagrams.validate_diagram_spec({"kind": "pie_chart"}, source_text="anything")
    assert ok is False


def test_validate_diagram_spec_rejects_missing_required_field():
    ok, reason = diagrams.validate_diagram_spec({"kind": "number_line", "min": 1}, source_text="1 to 10")
    assert ok is False
    assert "max" in reason


def test_validate_diagram_spec_rejects_badly_typed_required_field():
    ok, _ = diagrams.validate_diagram_spec({"kind": "number_line", "min": 1, "max": "ten"}, source_text="1 ten")
    assert ok is False


def test_validate_diagram_spec_rejects_ungrounded_number():
    spec = {"kind": "bar_compare", "labels": ["A", "B"], "values": [10, 99]}
    ok, reason = diagrams.validate_diagram_spec(spec, source_text="A scored 10 points, B scored 20 points.")
    assert ok is False
    assert "99" in reason


def test_validate_diagram_spec_accepts_grounded_spec():
    spec = {"kind": "bar_compare", "labels": ["A", "B"], "values": [10, 20]}
    ok, reason = diagrams.validate_diagram_spec(spec, source_text="A scored 10 points, B scored 20 points.")
    assert ok is True
    assert reason == ""


def test_validate_diagram_spec_ignores_single_digit_numbers_for_grounding():
    # Row counts / indices are too common to require literal grounding.
    spec = {"kind": "triangular_dots", "rows": 4, "label": "T4"}
    ok, _ = diagrams.validate_diagram_spec(spec, source_text="completely unrelated text")
    assert ok is True


# -- render_diagram: one smoke test per kind ------------------------------------

def test_render_diagram_number_line(tmp_path):
    out = diagrams.render_diagram({"kind": "number_line", "min": 1, "max": 10, "highlight": [5]}, 400, 300, tmp_path / "d.png")
    assert out.exists()
    assert Image.open(out).size == (400, 300)


def test_render_diagram_triangular_dots(tmp_path):
    out = diagrams.render_diagram({"kind": "triangular_dots", "rows": 4, "label": "T4 = 10"}, 400, 300, tmp_path / "d.png")
    assert out.exists()


def test_render_diagram_sequence_boxes(tmp_path):
    out = diagrams.render_diagram({"kind": "sequence_boxes", "items": ["1", "3", "6", "10"], "highlight_index": 2}, 400, 300, tmp_path / "d.png")
    assert out.exists()


def test_render_diagram_bar_compare(tmp_path):
    out = diagrams.render_diagram({"kind": "bar_compare", "labels": ["A", "B"], "values": [10, 20]}, 400, 300, tmp_path / "d.png")
    assert out.exists()


def test_render_diagram_shape_progression(tmp_path):
    out = diagrams.render_diagram({"kind": "shape_progression", "shape": "square", "counts": [1, 4, 9]}, 400, 300, tmp_path / "d.png")
    assert out.exists()


def test_render_diagram_trend_line(tmp_path):
    out = diagrams.render_diagram(
        {"kind": "trend_line", "labels": ["FY21", "FY22", "FY23"], "values": [100, 250, 412]}, 400, 300, tmp_path / "d.png"
    )
    assert out.exists()
    assert Image.open(out).size == (400, 300)


def test_render_diagram_raises_on_unknown_kind(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        diagrams.render_diagram({"kind": "pie_chart"}, 400, 300, tmp_path / "d.png")


# -- long multi-word items/labels must shrink-to-fit, not overflow into -------
# neighbouring boxes/bars (the actual bug behind a real video's "junk
# overlapping clumsy mathematical data" -- sequence_boxes was used with
# category names like "Triangular numbers" instead of short numbers, and the
# old fixed box-width-derived font size drew each label far wider than its
# own box, smearing across its neighbours).

def test_fit_text_shrinks_long_text_to_fit_max_width():
    img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(img)
    font, text = diagrams._fit_text(draw, "Triangular numbers", max_width=300, start_size=40)
    assert text == "Triangular numbers"  # shrunk, not truncated
    assert draw.textbbox((0, 0), text, font=font)[2] <= 300


def test_fit_text_truncates_with_ellipsis_when_shrinking_alone_cant_fit():
    img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(img)
    font, text = diagrams._fit_text(draw, "Triangular numbers", max_width=30, start_size=40, min_size=11)
    assert text.endswith("…")
    assert draw.textbbox((0, 0), text, font=font)[2] <= 30


def test_fit_text_sanitizes_unicode_whitespace_in_llm_authored_labels():
    # Diagram labels/values come straight from the LLM's diagram JSON -- same
    # tofu-box risk as on_screen_text (narrow no-break space etc. with no
    # glyph in the render font), so _fit_text must sanitize too.
    img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(img)
    font, text = diagrams._fit_text(draw, "Price 92", max_width=300, start_size=40)
    assert text == "Price 92"



def test_render_sequence_boxes_with_long_category_names_stays_within_each_box(tmp_path):
    out = diagrams.render_diagram(
        {
            "kind": "sequence_boxes",
            "items": ["Counting numbers", "Odd numbers", "Even numbers", "Triangular numbers"],
            "highlight_index": None,
        },
        800, 600, tmp_path / "d.png",
    )
    img = Image.open(out).convert("RGB")
    width, height = 800, 600
    n = 4
    box_w = min(int(width * 0.8 / n), int(width * 0.18))
    gap = int(box_w * 0.3)
    total_w = n * box_w + (n - 1) * gap
    x0 = (width - total_w) // 2
    y_mid = height // 2
    # The strip between two boxes must never show the white label text --
    # that would mean a label bled out of its own box into the gap.
    for i in range(n - 1):
        gap_x0 = x0 + i * (box_w + gap) + box_w
        for x in range(gap_x0 + 2, gap_x0 + gap - 2):
            assert img.getpixel((x, y_mid)) != (255, 255, 255)


# -- diagrams must stay clear of textcard's caption bar (0.68*height down) ------
# otherwise a diagram scene's own baked-in labels visually overlap the
# separate on_screen_text caption overlaid on top of it afterwards.

def _assert_nothing_drawn_below_caption_bar(path, width, height):
    img = Image.open(path).convert("RGB")
    caption_y0 = int(height * 0.68)
    for y in range(caption_y0, height):
        for x in range(0, width, 5):  # sample every 5th pixel, plenty for a solid-fill bug
            assert img.getpixel((x, y)) == diagrams.BG_COLOR, (
                f"diagram drew non-background pixel at ({x}, {y}), inside the caption band"
            )


def test_render_diagram_bar_compare_avoids_caption_band(tmp_path):
    out = diagrams.render_diagram(
        {"kind": "bar_compare", "labels": ["A", "B"], "values": [10, 20]}, 400, 300, tmp_path / "d.png"
    )
    _assert_nothing_drawn_below_caption_bar(out, 400, 300)


def test_render_diagram_triangular_dots_with_many_rows_avoids_caption_band(tmp_path):
    out = diagrams.render_diagram(
        {"kind": "triangular_dots", "rows": 10, "label": "T10 = 55"}, 400, 300, tmp_path / "d.png"
    )
    _assert_nothing_drawn_below_caption_bar(out, 400, 300)


def test_render_sequence_boxes_with_eight_items_stays_within_frame_width(tmp_path):
    # Regression: box_w was sized off box count alone (width*0.8/n) and the
    # inter-box gaps were added on top afterwards, so n=8 (the max allowed)
    # overflowed the frame by ~10px on each side -- visibly clipping the
    # first/last box in a real generated video's frame.
    out = diagrams.render_diagram(
        {"kind": "sequence_boxes", "items": [str(i) for i in range(8)], "highlight_index": None},
        1280, 720, tmp_path / "d.png",
    )
    img = Image.open(out).convert("RGB")
    n = 8
    box_w = min(int(1280 * 0.8 / (n + 0.3 * (n - 1))), int(1280 * 0.18))
    gap = int(box_w * 0.3)
    total_w = n * box_w + (n - 1) * gap
    x0 = (1280 - total_w) // 2
    assert x0 >= 0
    assert x0 + total_w <= 1280


def test_render_diagram_shape_progression_avoids_caption_band(tmp_path):
    out = diagrams.render_diagram(
        {"kind": "shape_progression", "shape": "square", "counts": [1, 4, 9, 16, 25]}, 400, 300, tmp_path / "d.png"
    )
    _assert_nothing_drawn_below_caption_bar(out, 400, 300)


def test_render_diagram_trend_line_avoids_caption_band(tmp_path):
    out = diagrams.render_diagram(
        {"kind": "trend_line", "labels": ["FY21", "FY22", "FY23"], "values": [100, 250, 412]}, 400, 300, tmp_path / "d.png"
    )
    _assert_nothing_drawn_below_caption_bar(out, 400, 300)


def test_validate_diagram_spec_accepts_trend_line():
    spec = {"kind": "trend_line", "labels": ["FY21", "FY22"], "values": [100, 250]}
    ok, reason = diagrams.validate_diagram_spec(spec, source_text="Revenue grew from 100 Cr in FY21 to 250 Cr in FY22.")
    assert ok is True
    assert reason == ""


def test_validate_diagram_spec_rejects_trend_line_with_ungrounded_number():
    spec = {"kind": "trend_line", "labels": ["FY21", "FY22"], "values": [100, 999]}
    ok, reason = diagrams.validate_diagram_spec(spec, source_text="Revenue grew from 100 Cr in FY21 to 250 Cr in FY22.")
    assert ok is False
    assert "999" in reason
