from pipeline import chat_prefs


def test_get_voice_defaults_to_empty_string(isolated_dirs, monkeypatch):
    cache_dir, _ = isolated_dirs
    monkeypatch.setattr(chat_prefs, "_PREFS_PATH", cache_dir / "chat_prefs.json")
    assert chat_prefs.get_voice(12345) == ""


def test_set_and_get_voice_round_trips(isolated_dirs, monkeypatch):
    cache_dir, _ = isolated_dirs
    monkeypatch.setattr(chat_prefs, "_PREFS_PATH", cache_dir / "chat_prefs.json")

    chat_prefs.set_voice(555, "hi-IN-SwaraNeural")
    assert chat_prefs.get_voice(555) == "hi-IN-SwaraNeural"
    # chat_id passed as int vs str should resolve to the same stored pref.
    assert chat_prefs.get_voice("555") == "hi-IN-SwaraNeural"


def test_set_voice_does_not_clobber_other_chats(isolated_dirs, monkeypatch):
    cache_dir, _ = isolated_dirs
    monkeypatch.setattr(chat_prefs, "_PREFS_PATH", cache_dir / "chat_prefs.json")

    chat_prefs.set_voice(1, "ta-IN-PallaviNeural")
    chat_prefs.set_voice(2, "hi-IN-SwaraNeural")

    assert chat_prefs.get_voice(1) == "ta-IN-PallaviNeural"
    assert chat_prefs.get_voice(2) == "hi-IN-SwaraNeural"


def test_load_survives_corrupt_json(isolated_dirs, monkeypatch):
    cache_dir, _ = isolated_dirs
    prefs_path = cache_dir / "chat_prefs.json"
    prefs_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(chat_prefs, "_PREFS_PATH", prefs_path)

    assert chat_prefs.get_voice(1) == ""
