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
    started = []
    # Replace the background-thread launcher with a synchronous stand-in so the
    # test doesn't depend on timing, and doesn't touch the real pipeline.
    monkeypatch.setattr(
        telegram_bot.threading, "Thread",
        lambda target, args, daemon: mock.Mock(start=lambda: started.append((target, args))),
    )
    telegram_bot.handle_update({"message": {"chat": {"id": 1}, "text": "5 facts about octopuses"}})
    assert len(started) == 1
    assert started[0][0] is telegram_bot._run_job
    assert started[0][1] == (1, "5 facts about octopuses")


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
