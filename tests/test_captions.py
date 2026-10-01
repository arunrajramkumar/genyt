from pipeline import captions


def test_format_ts_basic():
    assert captions._format_ts(0) == "00:00:00,000"
    assert captions._format_ts(1.5) == "00:00:01,500"
    assert captions._format_ts(61) == "00:01:01,000"
    assert captions._format_ts(3661.234) == "01:01:01,234"


def test_write_srt_formats_sequential_blocks(tmp_path):
    scenes = [
        {"narration": "Hello world.", "start": 0.0, "end": 2.5},
        {"narration": "Second line here.", "start": 2.5, "end": 5.0},
    ]
    out_path = captions.write_srt(scenes, tmp_path / "out.srt")

    text = out_path.read_text(encoding="utf-8")
    assert text.splitlines() == [
        "1",
        "00:00:00,000 --> 00:00:02,500",
        "Hello world.",
        "",
        "2",
        "00:00:02,500 --> 00:00:05,000",
        "Second line here.",
    ]


def test_write_srt_empty_scenes(tmp_path):
    out_path = captions.write_srt([], tmp_path / "empty.srt")
    assert out_path.read_text(encoding="utf-8") == ""
