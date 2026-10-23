import json
from unittest import mock

import pytest

from pipeline import tts


@pytest.mark.parametrize("raw,expected_fragment", [
    ("XYZ Ltd.", "XYZ Limited"),
    ("ABC Pvt Ltd", "ABC Private Limited"),
    ("P/E ratio is 22", "price to earnings ratio is 22"),
    ("Revenue grew 18% YoY", "18% year over year"),
    ("Rs 412 crore", "412 crore rupees"),
    ("₹412 crore", "412 crore rupees"),
    ("ROE of 22% Cr", "return on equity of 22% crore"),
    ("ROCE stood at 18%", "return on capital employed stood at 18%"),
    ("CAGR of 15%", "compound annual growth rate of 15%"),
    ("EBITDA margin improved", "earnings before interest, tax, depreciation and amortization margin improved"),
    ("NPA ratio fell", "non-performing assets ratio fell"),
    ("CASA ratio rose", "current account savings account ratio rose"),
    ("NIM expanded", "net interest margin expanded"),
    ("Revenue grew to ~30 crore this year.", "approximately 30 crore"),
])
def test_normalize_for_speech_expands_abbreviations(raw, expected_fragment):
    assert expected_fragment in tts._normalize_for_speech(raw)


def test_normalize_for_speech_leaves_plain_text_untouched():
    plain = "This is a normal sentence with no abbreviations."
    assert tts._normalize_for_speech(plain) == plain


@pytest.mark.parametrize("raw", [
    "The company completed its Incorporation in 2010.",
    "Increase in income was incredible, says the incorporated firm.",
    "The Corporation announced record profits.",
    "This is a Corporate strategy.",
])
def test_normalize_for_speech_does_not_mangle_words_containing_abbreviation_as_prefix(raw):
    """Regression guard: 'Inc'/'Corp' etc. patterns used to have only a leading
    \\b with no trailing boundary, so they matched as a bare prefix of longer
    words like "Incorporation"/"Corporation" and spliced the expansion into
    the leftover letters, producing garbage like "Incorporatedorporation"."""
    assert tts._normalize_for_speech(raw) == raw


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


def test_synthesize_retries_on_raw_aiohttp_websocket_handshake_error(tmp_path, monkeypatch):
    # Regression for a real GitHub Actions failure: a sustained 403 block
    # exhausts edge-tts's own single built-in retry and leaks a raw
    # aiohttp.WSServerHandshakeError (not an EdgeTTSException subclass) --
    # our retry loop must still catch it instead of crashing the whole run.
    monkeypatch.setattr(tts, "FFPROBE_BIN", "/usr/bin/ffprobe")
    monkeypatch.setattr(tts.time, "sleep", lambda s: None)
    attempts = {"count": 0}

    async def flaky_synthesize_async(text, voice, out_path):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise tts.aiohttp.WSServerHandshakeError(
                request_info=mock.Mock(), history=(), status=403, message="Invalid response status"
            )
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


def test_synthesize_uses_google_when_engine_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "FFPROBE_BIN", "/usr/bin/ffprobe")
    monkeypatch.setattr(tts.config, "TTS_ENGINE", "google")
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_USAGE_FILE", tmp_path / "usage.json")
    monkeypatch.setattr(tts, "_probe_duration", lambda path: 3.0)
    calls = []

    def fake_synthesize_google(text, out_path, voice=None):
        calls.append(text)
        out_path.write_bytes(b"fake-mp3")

    monkeypatch.setattr(tts, "_synthesize_google", fake_synthesize_google)

    duration = tts.synthesize("hello", tmp_path / "out.mp3")
    assert calls == ["hello"]
    assert duration == 3.0


def test_synthesize_retries_google_on_request_exception_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "FFPROBE_BIN", "/usr/bin/ffprobe")
    monkeypatch.setattr(tts.config, "TTS_ENGINE", "google")
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_USAGE_FILE", tmp_path / "usage.json")
    monkeypatch.setattr(tts.time, "sleep", lambda s: None)
    monkeypatch.setattr(tts, "_probe_duration", lambda path: 3.0)
    attempts = {"count": 0}

    def flaky_synthesize_google(text, out_path, voice=None):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise tts.requests.RequestException("rate limited")
        out_path.write_bytes(b"fake-mp3")

    monkeypatch.setattr(tts, "_synthesize_google", flaky_synthesize_google)

    duration = tts.synthesize("hello", tmp_path / "out.mp3")
    assert attempts["count"] == 3
    assert duration == 3.0


def test_synthesize_raises_when_google_monthly_cap_would_be_exceeded(tmp_path, monkeypatch):
    usage_file = tmp_path / "usage.json"
    usage_file.write_text('{"month": "2026-10", "characters": 399990}')
    monkeypatch.setattr(tts, "FFPROBE_BIN", "/usr/bin/ffprobe")
    monkeypatch.setattr(tts.config, "TTS_ENGINE", "google")
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_USAGE_FILE", usage_file)
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_MONTHLY_CHAR_CAP", 400000)
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_SAFETY_BUFFER_CHARS", 0)
    monkeypatch.setattr(tts, "_current_month", lambda: "2026-10")

    def fail_if_called(text, out_path, voice=None):
        raise AssertionError("must not call the API once the cap would be exceeded")

    monkeypatch.setattr(tts, "_synthesize_google", fail_if_called)

    with pytest.raises(tts.TTSQuotaExceededError, match="cap"):
        tts.synthesize("x" * 20, tmp_path / "out.mp3")


def test_ensure_google_tts_budget_raises_when_cap_would_be_exceeded(tmp_path, monkeypatch):
    usage_file = tmp_path / "usage.json"
    usage_file.write_text('{"month": "2026-10", "characters": 399990}')
    monkeypatch.setattr(tts.config, "TTS_ENGINE", "google")
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_USAGE_FILE", usage_file)
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_MONTHLY_CHAR_CAP", 400000)
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_SAFETY_BUFFER_CHARS", 0)
    monkeypatch.setattr(tts, "_current_month", lambda: "2026-10")

    with pytest.raises(tts.TTSQuotaExceededError, match="cap"):
        tts.ensure_google_tts_budget(20)


def test_ensure_google_tts_budget_respects_safety_buffer(tmp_path, monkeypatch):
    usage_file = tmp_path / "usage.json"
    usage_file.write_text('{"month": "2026-10", "characters": 394999}')
    monkeypatch.setattr(tts.config, "TTS_ENGINE", "google")
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_USAGE_FILE", usage_file)
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_MONTHLY_CHAR_CAP", 400000)
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_SAFETY_BUFFER_CHARS", 5000)
    monkeypatch.setattr(tts, "_current_month", lambda: "2026-10")

    # 394999 + 2 = 395001, which is within the raw 400000 cap but inside the
    # 5000-char safety buffer (usable cap is 395000) -- should still raise.
    with pytest.raises(tts.TTSQuotaExceededError, match="cap"):
        tts.ensure_google_tts_budget(2)


def test_ensure_google_tts_budget_is_a_noop_for_non_google_engine(monkeypatch):
    monkeypatch.setattr(tts.config, "TTS_ENGINE", "edge")
    tts.ensure_google_tts_budget(10 ** 9)  # must not raise regardless of size


def test_synthesize_records_google_usage_and_resets_on_new_month(tmp_path, monkeypatch):
    usage_file = tmp_path / "usage.json"
    usage_file.write_text('{"month": "2025-01", "characters": 399999}')  # stale month
    monkeypatch.setattr(tts, "FFPROBE_BIN", "/usr/bin/ffprobe")
    monkeypatch.setattr(tts.config, "TTS_ENGINE", "google")
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_USAGE_FILE", usage_file)
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_MONTHLY_CHAR_CAP", 400000)
    monkeypatch.setattr(tts, "_current_month", lambda: "2026-10")
    monkeypatch.setattr(tts, "_probe_duration", lambda path: 1.0)
    monkeypatch.setattr(tts, "_synthesize_google", lambda text, out_path, voice=None: out_path.write_bytes(b"fake-mp3"))

    tts.synthesize("hello", tmp_path / "out.mp3")

    usage = json.loads(usage_file.read_text())
    assert usage == {"month": "2026-10", "characters": len("hello")}


# ---- premium (Chirp 3 HD) voice first, Neural2 after its cap ----

PREMIUM = "en-IN-Chirp3-HD-Charon"


@pytest.fixture
def google_premium(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "FFPROBE_BIN", "/usr/bin/ffprobe")
    monkeypatch.setattr(tts.config, "TTS_ENGINE", "google")
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_PREMIUM_VOICE", PREMIUM)
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_VOICE", "en-IN-Neural2-B")
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_USAGE_FILE", tmp_path / "neural.json")
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_PREMIUM_USAGE_FILE", tmp_path / "chirp.json")
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_PREMIUM_MONTHLY_CHAR_CAP", 10_000)
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_SAFETY_BUFFER_CHARS", 100)
    monkeypatch.setattr(tts, "_probe_duration", lambda path: 1.0)
    voices = []

    def fake(text, out_path, voice=None):
        voices.append(voice)
        out_path.write_bytes(b"mp3")
    monkeypatch.setattr(tts, "_synthesize_google", fake)
    return tmp_path, voices


def _chars(path):
    return json.loads(path.read_text())["characters"] if path.exists() else 0


def test_premium_voice_used_and_counted_separately(google_premium):
    tmp, voices = google_premium
    tts.ensure_google_tts_budget(10)
    tts.synthesize("hello there", tmp / "a.mp3")
    assert voices == [PREMIUM]
    assert _chars(tmp / "chirp.json") == 11 and _chars(tmp / "neural.json") == 0


def test_falls_back_to_neural2_once_premium_cap_is_reached(google_premium):
    tmp, voices = google_premium
    (tmp / "chirp.json").write_text(json.dumps({"month": tts._current_month(), "characters": 9_895}))
    tts.ensure_google_tts_budget(10)  # 9,895 + 10 > 10,000 - 100
    tts.synthesize("hello there", tmp / "a.mp3")
    assert voices == ["en-IN-Neural2-B"] and _chars(tmp / "neural.json") == 11


def test_premium_counter_resets_each_month(google_premium):
    tmp, voices = google_premium
    (tmp / "chirp.json").write_text(json.dumps({"month": "2020-01", "characters": 9_999}))
    tts.ensure_google_tts_budget(10)
    tts.synthesize("hi", tmp / "a.mp3")
    assert voices == [PREMIUM]


def test_one_video_never_mixes_voices(google_premium):
    tmp, voices = google_premium
    tts.ensure_google_tts_budget(30)
    for i in range(3):
        tts.synthesize("scene text", tmp / f"{i}.mp3")
    assert voices == [PREMIUM] * 3


def test_rejected_premium_voice_falls_back_for_the_rest_of_the_run(google_premium, monkeypatch):
    tmp, voices = google_premium

    def fake(text, out_path, voice=None):
        voices.append(voice)
        if voice == PREMIUM:
            raise tts._VoiceRejectedError("invalid voice name")
        out_path.write_bytes(b"mp3")
    monkeypatch.setattr(tts, "_synthesize_google", fake)
    tts.ensure_google_tts_budget(10)
    tts.synthesize("hello", tmp / "a.mp3")
    tts.ensure_google_tts_budget(10)  # next video: premium not retried
    tts.synthesize("hello", tmp / "b.mp3")
    assert voices == [PREMIUM, "en-IN-Neural2-B", "en-IN-Neural2-B"]


def test_neural2_cap_still_stops_the_run_after_premium_is_used_up(google_premium, monkeypatch):
    tmp, voices = google_premium
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_MONTHLY_CHAR_CAP", 10_000)
    for f in ("chirp.json", "neural.json"):
        (tmp / f).write_text(json.dumps({"month": tts._current_month(), "characters": 9_950}))
    with pytest.raises(tts.TTSQuotaExceededError):
        tts.ensure_google_tts_budget(100)
    assert voices == []


def test_premium_off_when_unset(google_premium, monkeypatch):
    tmp, voices = google_premium
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_PREMIUM_VOICE", "")
    tts.ensure_google_tts_budget(10)
    tts.synthesize("hi", tmp / "a.mp3")
    assert voices == ["en-IN-Neural2-B"] and not (tmp / "chirp.json").exists()


def test_premium_rejection_is_detected_from_http_400(monkeypatch, tmp_path):
    monkeypatch.setattr(tts.config, "GOOGLE_TTS_PREMIUM_VOICE", PREMIUM)
    resp = mock.Mock(status_code=400, text='{"error": "voice not found"}')
    monkeypatch.setattr(tts.requests, "post", lambda *a, **k: resp)
    with pytest.raises(tts._VoiceRejectedError):
        tts._synthesize_google("hi", tmp_path / "a.mp3", PREMIUM)


@pytest.mark.parametrize("text, spoken", [
    ("profit of ₹1,726 crore", "profit of 1,726 crore rupees"),
    ("price band is ₹660 to ₹693 per share", "price band is 660 to 693 rupees per share"),
    ("a ₹500-crore fresh issue", "a 500 crore rupees fresh issue"),
    ("AUM of ₹5.85 lakh crore", "assets under management of 5.85 lakh crore rupees"),
    ("₹660–693", "660 to 693 rupees"),
    ("P/B is 4.6x, down 12 bps", "price to book is 4.6 times, down 12 basis points"),
    ("Q1 FY27 and FY2026", "Q1 F Y 27 and F Y 2026"),
    ("NII rose", "net interest income rose"),
    ("valued at 171x earnings", "valued at 171 times earnings"),
])
def test_amounts_and_finance_terms_read_naturally(text, spoken):
    assert tts._normalize_for_speech(text) == spoken


def test_words_ending_in_x_are_untouched():
    assert tts._normalize_for_speech("Max Healthcare and 3M index") == "Max Healthcare and 3M index"
