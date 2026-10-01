from unittest import mock

import pytest

from pipeline import tts


@pytest.mark.parametrize("raw,expected_fragment", [
    ("XYZ Ltd.", "XYZ Limited"),
    ("ABC Pvt Ltd", "ABC Private Limited"),
    ("P/E ratio is 22", "price to earnings ratio is 22"),
    ("Revenue grew 18% YoY", "18% year over year"),
    ("Rs 412 crore", "Rupees 412 crore"),
    ("ROE of 22% Cr", "ROE of 22% Crore"),
])
def test_normalize_for_speech_expands_abbreviations(raw, expected_fragment):
    assert expected_fragment in tts._normalize_for_speech(raw)


def test_normalize_for_speech_leaves_plain_text_untouched():
    plain = "This is a normal sentence with no abbreviations."
    assert tts._normalize_for_speech(plain) == plain


def test_synthesize_raises_without_ffprobe(monkeypatch):
    monkeypatch.setattr(tts, "FFPROBE_BIN", None)
    with pytest.raises(RuntimeError, match="ffprobe"):
        tts.synthesize("hello", __import__("pathlib").Path("/tmp/out.mp3"))


def test_synthesize_passes_override_voice_not_config_default(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "FFPROBE_BIN", "/usr/bin/ffprobe")
    captured = {}

    async def fake_synthesize_async(text, voice, out_path):
        captured["voice"] = voice
        out_path.write_bytes(b"fake-mp3")

    monkeypatch.setattr(tts, "_synthesize_async", fake_synthesize_async)
    monkeypatch.setattr(tts, "_probe_duration", lambda path: 3.5)

    duration = tts.synthesize("hello", tmp_path / "out.mp3", voice="hi-IN-SwaraNeural")
    assert captured["voice"] == "hi-IN-SwaraNeural"
    assert duration == 3.5


def test_synthesize_defaults_to_config_voice_when_none_given(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "FFPROBE_BIN", "/usr/bin/ffprobe")
    monkeypatch.setattr(tts.config, "TTS_VOICE", "en-IN-PrabhatNeural")
    captured = {}

    async def fake_synthesize_async(text, voice, out_path):
        captured["voice"] = voice
        out_path.write_bytes(b"fake-mp3")

    monkeypatch.setattr(tts, "_synthesize_async", fake_synthesize_async)
    monkeypatch.setattr(tts, "_probe_duration", lambda path: 1.0)

    tts.synthesize("hello", tmp_path / "out.mp3")
    assert captured["voice"] == "en-IN-PrabhatNeural"


def test_synthesize_retries_on_edge_tts_exception_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "FFPROBE_BIN", "/usr/bin/ffprobe")
    monkeypatch.setattr(tts.time, "sleep", lambda s: None)
    attempts = {"count": 0}

    async def flaky_synthesize_async(text, voice, out_path):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise tts.edge_tts.exceptions.NoAudioReceived("no audio")
        out_path.write_bytes(b"fake-mp3")

    monkeypatch.setattr(tts, "_synthesize_async", flaky_synthesize_async)
    monkeypatch.setattr(tts, "_probe_duration", lambda path: 2.0)

    duration = tts.synthesize("hello", tmp_path / "out.mp3")
    assert attempts["count"] == 3
    assert duration == 2.0


def test_synthesize_gives_up_after_max_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "FFPROBE_BIN", "/usr/bin/ffprobe")
    monkeypatch.setattr(tts.time, "sleep", lambda s: None)

    async def always_fails(text, voice, out_path):
        raise tts.edge_tts.exceptions.NoAudioReceived("no audio")

    monkeypatch.setattr(tts, "_synthesize_async", always_fails)

    with pytest.raises(tts.edge_tts.exceptions.NoAudioReceived):
        tts.synthesize("hello", tmp_path / "out.mp3")
