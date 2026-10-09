from pipeline.segmented_script import parse_segments
from scripts.build_education_queue import build_theme_entry, group_sections


_GROUP = [
    {"id": "1.1", "title": "Triangular Numbers", "body": "Triangular numbers are 1, 3, 6, 10, 15 formed by stacking dots in rows."},
    {"id": "1.2", "title": "Square Numbers", "body": "Square numbers are 1, 4, 9, 16 formed by arranging dots in a square grid."},
]


def test_build_theme_entry_emits_at_least_three_timed_headers():
    entry = build_theme_entry("Grade 6 CBSE Maths", 1, "Patterns in Mathematics", 1, 1, _GROUP, ["NCERT", "CBSE"], "CBSE | Grade 6 Maths")
    segments = parse_segments(entry["theme"])
    assert len(segments) >= 4  # Hook + 2 sections + Recap
    labels = [s["label"] for s in segments]
    assert labels[0] == "Hook"
    assert labels[-1] == "Recap"
    assert "Triangular Numbers" in labels
    assert "Square Numbers" in labels


def test_build_theme_entry_section_segments_carry_the_real_textbook_facts():
    entry = build_theme_entry("Grade 6 CBSE Maths", 1, "Patterns in Mathematics", 1, 1, _GROUP, [], None)
    segments = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}
    assert "1, 3, 6, 10, 15" in segments["Triangular Numbers"]
    assert "1, 4, 9, 16" in segments["Square Numbers"]


def test_build_theme_entry_hook_and_recap_have_no_fabricated_facts_required():
    entry = build_theme_entry("Grade 6 CBSE Maths", 1, "Patterns in Mathematics", 1, 1, _GROUP, [], None)
    segments = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}
    assert "Triangular Numbers & Square Numbers" in segments["Hook"]
    assert "Triangular Numbers & Square Numbers" in segments["Recap"]


def test_build_theme_entry_duration_sec_matches_sum_of_segment_durations():
    entry = build_theme_entry("Grade 6 CBSE Maths", 1, "Patterns in Mathematics", 1, 1, _GROUP, [], None)
    segments = parse_segments(entry["theme"])
    assert sum(s["duration"] for s in segments) == entry["duration_sec"]


def test_build_theme_entry_preserves_extra_tags_and_thumbnail_badge():
    entry = build_theme_entry("Grade 6 CBSE Maths", 1, "Patterns in Mathematics", 1, 1, _GROUP, ["NCERT", "CBSE"], "CBSE | Grade 6 Maths")
    assert entry["extra_tags"][:2] == ["NCERT", "CBSE"]
    assert entry["thumbnail_badge"] == "CBSE | Grade 6 Maths"


def test_build_theme_entry_single_section_still_produces_three_headers():
    single = [_GROUP[0]]
    entry = build_theme_entry("Grade 6 CBSE Maths", 1, "Patterns in Mathematics", 1, 1, single, [], None)
    segments = parse_segments(entry["theme"])
    assert len(segments) == 3  # Hook + 1 section + Recap


def test_group_sections_splits_evenly_by_max_sections_per_part():
    sections = [{"id": f"1.{i}", "title": f"S{i}", "body": "x"} for i in range(1, 5)]
    groups = group_sections(sections)
    assert sum(len(g) for g in groups) == 4
    assert all(len(g) <= 2 for g in groups)
