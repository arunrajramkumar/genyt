from PIL import Image

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


def test_render_diagram_raises_on_unknown_kind(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        diagrams.render_diagram({"kind": "pie_chart"}, 400, 300, tmp_path / "d.png")
