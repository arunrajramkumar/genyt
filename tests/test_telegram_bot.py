from unittest import mock

import pytest

from pipeline import telegram_bot, chat_prefs


# -- _resolve_voice -----------------------------------------------------------

@pytest.mark.parametrize("alias,expected", [
    ("hindi", "hi-IN-SwaraNeural"),
    ("Hindi", "hi-IN-SwaraNeural"),
    ("  tamil  ", "ta-IN-PallaviNeural"),
    ("ENGLISH", "en-IN-PrabhatNeural"),
    ("english-us", "en-US-AriaNeural"),
])
def test_resolve_voice_aliases(alias, expected):
    assert telegram_bot._resolve_voice(alias) == expected


def test_resolve_voice_accepts_raw_voice_id():
    assert telegram_bot._resolve_voice("ta-IN-ValluvarNeural") == "ta-IN-ValluvarNeural"


@pytest.mark.parametrize("bad", ["nonsense", "", "hindi-ish", "EN-US-AriaNeural", "en-us-aria"])
def test_resolve_voice_rejects_unknown_input(bad):
    assert telegram_bot._resolve_voice(bad) == ""


# -- handle_update ------------------------------------------------------------

@pytest.fixture
def sent_messages(monkeypatch):
    sent = []
    monkeypatch.setattr(telegram_bot, "_send_message", lambda chat_id, text: sent.append((chat_id, text)))
    return sent


@pytest.fixture
def isolated_prefs(isolated_dirs, monkeypatch):
    cache_dir, _ = isolated_dirs
    monkeypatch.setattr(chat_prefs, "_PREFS_PATH", cache_dir / "chat_prefs.json")


@pytest.fixture(autouse=True)
def _clear_pending_prompt_state():
    telegram_bot._pending_fragments.clear()
    telegram_bot._pending_timers.clear()
    yield
    telegram_bot._pending_fragments.clear()
    telegram_bot._pending_timers.clear()


class _ImmediateTimer:
    """Stand-in for threading.Timer that fires synchronously on start(), for
    tests that only send one message and don't care about the debounce delay."""

    def __init__(self, interval, function, args=()):
        self.function = function
        self.args = args

    def start(self):
        self.function(*self.args)

    def cancel(self):
        pass


class _ManualTimer:
    """Stand-in for threading.Timer that never fires on its own — tests fire
    it explicitly to simulate the debounce window elapsing, so multi-message
    coalescing can be tested deterministically without sleeping."""

    instances = []

    def __init__(self, interval, function, args=()):
        self.function = function
        self.args = args
        self.cancelled = False
        _ManualTimer.instances.append(self)

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            self.function(*self.args)


def test_handle_update_ignores_messages_without_text_or_chat(sent_messages, isolated_prefs):
    telegram_bot.handle_update({})
    telegram_bot.handle_update({"message": {"chat": {"id": 1}, "text": ""}})
    telegram_bot.handle_update({"message": {"chat": {}, "text": "hello"}})
    assert sent_messages == []


def test_handle_update_help_and_start(sent_messages, isolated_prefs):
    telegram_bot.handle_update({"message": {"chat": {"id": 1}, "text": "/start"}})
    telegram_bot.handle_update({"message": {"chat": {"id": 1}, "text": "/help"}})
    assert len(sent_messages) == 2
    assert all(text == telegram_bot.HELP_TEXT for _, text in sent_messages)


def test_handle_update_voice_query_shows_default_when_unset(sent_messages, isolated_prefs):
    telegram_bot.handle_update({"message": {"chat": {"id": 1}, "text": "/voice"}})
    assert len(sent_messages) == 1
    chat_id, text = sent_messages[0]
    assert "Current voice:" in text
    from pipeline import config
    assert config.TTS_VOICE in text


def test_handle_update_voice_set_persists_and_confirms(sent_messages, isolated_prefs):
    telegram_bot.handle_update({"message": {"chat": {"id": 42}, "text": "/voice hindi"}})
    assert chat_prefs.get_voice(42) == "hi-IN-SwaraNeural"
    assert "hi-IN-SwaraNeural" in sent_messages[-1][1]

    telegram_bot.handle_update({"message": {"chat": {"id": 42}, "text": "/voice"}})
    assert "hi-IN-SwaraNeural" in sent_messages[-1][1]


def test_handle_update_voice_set_rejects_unknown_language(sent_messages, isolated_prefs):
    telegram_bot.handle_update({"message": {"chat": {"id": 42}, "text": "/voice klingon"}})
    assert chat_prefs.get_voice(42) == ""
    assert "Didn't recognize" in sent_messages[-1][1]


def test_handle_update_blocks_unauthorized_chat(sent_messages, isolated_prefs, monkeypatch):
    monkeypatch.setattr(telegram_bot, "ALLOWED_CHAT_IDS", {"1"})
    telegram_bot.handle_update({"message": {"chat": {"id": 999}, "text": "hello"}})
    assert "not authorized" in sent_messages[-1][1]


def test_handle_update_allows_whitelisted_chat_and_kicks_off_job(sent_messages, isolated_prefs, monkeypatch):
    monkeypatch.setattr(telegram_bot, "ALLOWED_CHAT_IDS", {"1"})
    # Stand in for threading.Timer/Thread with synchronous equivalents so the
    # test doesn't depend on the real debounce delay or touch the real pipeline.
    monkeypatch.setattr(telegram_bot.threading, "Timer", _ImmediateTimer)
    started = []
    monkeypatch.setattr(
        telegram_bot.threading, "Thread",
        lambda target, args, daemon: mock.Mock(start=lambda: started.append((target, args))),
    )
    telegram_bot.handle_update({"message": {"chat": {"id": 1}, "text": "5 facts about octopuses"}})
    assert len(started) == 1
    assert started[0][0] is telegram_bot._run_job
    assert started[0][1] == (1, "5 facts about octopuses")


def test_handle_update_coalesces_rapid_fragments_into_one_job(sent_messages, isolated_prefs, monkeypatch):
    """Regression test: Telegram clients that send each line of a pasted
    multi-paragraph prompt as a separate message (e.g. Desktop's default
    Enter-sends-message behavior) must have those fragments joined back into
    one prompt, not each trigger its own tiny, nonsensical video."""
    _ManualTimer.instances.clear()
    monkeypatch.setattr(telegram_bot.threading, "Timer", _ManualTimer)
    started = []
    monkeypatch.setattr(
        telegram_bot.threading, "Thread",
        lambda target, args, daemon: mock.Mock(start=lambda: started.append((target, args))),
    )

    telegram_bot.handle_update({"message": {"chat": {"id": 5}, "text": "0-5 seconds - Hook"}})
    telegram_bot.handle_update({"message": {"chat": {"id": 5}, "text": "5-20 seconds - Body"}})

    # No job should have started yet — still inside the debounce window.
    assert started == []
    assert _ManualTimer.instances[0].cancelled is True  # superseded by the 2nd fragment
    assert _ManualTimer.instances[1].cancelled is False

    _ManualTimer.instances[1].fire()  # simulate the debounce window elapsing

    assert len(started) == 1
    assert started[0][0] is telegram_bot._run_job
    assert started[0][1] == (5, "0-5 seconds - Hook\n\n5-20 seconds - Body")


def test_handle_update_keeps_different_chats_independent(sent_messages, isolated_prefs, monkeypatch):
    _ManualTimer.instances.clear()
    monkeypatch.setattr(telegram_bot.threading, "Timer", _ManualTimer)
    started = []
    monkeypatch.setattr(
        telegram_bot.threading, "Thread",
        lambda target, args, daemon: mock.Mock(start=lambda: started.append((target, args))),
    )

    telegram_bot.handle_update({"message": {"chat": {"id": 1}, "text": "chat one prompt"}})
    telegram_bot.handle_update({"message": {"chat": {"id": 2}, "text": "chat two prompt"}})

    for timer in _ManualTimer.instances:
        timer.fire()

    assert len(started) == 2
    assert (1, "chat one prompt") in [s[1] for s in started]
    assert (2, "chat two prompt") in [s[1] for s in started]


def test_run_job_uses_stored_voice_preference(isolated_prefs, monkeypatch):
    chat_prefs.set_voice(7, "ta-IN-PallaviNeural")
    sent = []
    monkeypatch.setattr(telegram_bot, "_send_message", lambda chat_id, text: sent.append(text))
    monkeypatch.setattr(telegram_bot, "_send_video", lambda *a, **k: None)
    monkeypatch.setattr(telegram_bot, "_send_photo", lambda *a, **k: None)
    monkeypatch.setattr(telegram_bot, "_send_document", lambda *a, **k: None)

    fake_script = {"title": "T", "scenes": []}
    monkeypatch.setattr(telegram_bot.prompt_script, "write_script_from_prompt", lambda prompt, duration_sec: fake_script)

    captured = {}
    def fake_produce(script, orientation, on_progress, voice):
        captured["voice"] = voice
        return {"video": "v", "thumbnail": "t", "srt": "s", "metadata": "m"}
    monkeypatch.setattr(telegram_bot.producer, "produce_from_script", fake_produce)

    telegram_bot._run_job(7, "a prompt")

    assert captured["voice"] == "ta-IN-PallaviNeural"


def test_run_job_reports_failure_to_chat_instead_of_raising(isolated_prefs, monkeypatch):
    sent = []
    monkeypatch.setattr(telegram_bot, "_send_message", lambda chat_id, text: sent.append(text))
    monkeypatch.setattr(telegram_bot.prompt_script, "write_script_from_prompt",
                         mock.Mock(side_effect=RuntimeError("boom")))

    telegram_bot._run_job(1, "a prompt")  # must not raise

    assert any("failed" in t.lower() and "boom" in t for t in sent)


def test_run_job_announces_youtube_link_when_upload_succeeded(isolated_prefs, monkeypatch):
    sent = []
    monkeypatch.setattr(telegram_bot, "_send_message", lambda chat_id, text: sent.append(text))
    monkeypatch.setattr(telegram_bot, "_send_video", lambda *a, **k: None)
    monkeypatch.setattr(telegram_bot, "_send_photo", lambda *a, **k: None)
    monkeypatch.setattr(telegram_bot, "_send_document", lambda *a, **k: None)
    monkeypatch.setattr(telegram_bot.prompt_script, "write_script_from_prompt",
                         lambda prompt, duration_sec: {"title": "T", "scenes": []})
    monkeypatch.setattr(
        telegram_bot.producer, "produce_from_script",
        lambda script, orientation, on_progress, voice: {
            "video": "v", "thumbnail": "t", "srt": "s", "metadata": "m",
            "youtube_url": "https://youtu.be/abc123",
        },
    )

    telegram_bot._run_job(1, "a prompt")

    assert any("https://youtu.be/abc123" in t for t in sent)


def test_run_job_says_nothing_extra_when_upload_was_skipped(isolated_prefs, monkeypatch):
    sent = []
    monkeypatch.setattr(telegram_bot, "_send_message", lambda chat_id, text: sent.append(text))
    monkeypatch.setattr(telegram_bot, "_send_video", lambda *a, **k: None)
    monkeypatch.setattr(telegram_bot, "_send_photo", lambda *a, **k: None)
    monkeypatch.setattr(telegram_bot, "_send_document", lambda *a, **k: None)
    monkeypatch.setattr(telegram_bot.prompt_script, "write_script_from_prompt",
                         lambda prompt, duration_sec: {"title": "T", "scenes": []})
    monkeypatch.setattr(
        telegram_bot.producer, "produce_from_script",
        lambda script, orientation, on_progress, voice: {
            "video": "v", "thumbnail": "t", "srt": "s", "metadata": "m", "youtube_url": None,
        },
    )

    telegram_bot._run_job(1, "a prompt")

    assert not any("youtube" in t.lower() for t in sent)
