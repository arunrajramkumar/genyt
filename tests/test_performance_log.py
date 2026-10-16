import json

from pipeline import performance_log


def test_load_returns_empty_list_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(performance_log.config, "COMPANY_PERFORMANCE_LOG_FILE", tmp_path / "missing.json")
    assert performance_log.load() == []


def test_load_returns_empty_list_on_corrupt_json(monkeypatch, tmp_path):
    log_file = tmp_path / "log.json"
    log_file.write_text("not json")
    monkeypatch.setattr(performance_log.config, "COMPANY_PERFORMANCE_LOG_FILE", log_file)
    assert performance_log.load() == []


def test_record_appends_entry_with_video_id_symbol_and_company_name(monkeypatch, tmp_path):
    log_file = tmp_path / "log.json"
    monkeypatch.setattr(performance_log.config, "COMPANY_PERFORMANCE_LOG_FILE", log_file)

    performance_log.record("abc123", "CDSL", "CDSL Ltd")

    entries = json.loads(log_file.read_text())
    assert entries == [{"video_id": "abc123", "symbol": "CDSL", "company_name": "CDSL Ltd"}]


def test_record_accumulates_across_calls(monkeypatch, tmp_path):
    log_file = tmp_path / "log.json"
    monkeypatch.setattr(performance_log.config, "COMPANY_PERFORMANCE_LOG_FILE", log_file)

    performance_log.record("v1", "CDSL", "CDSL Ltd")
    performance_log.record("v2", "TCS", "Tata Consultancy")

    assert [e["video_id"] for e in performance_log.load()] == ["v1", "v2"]


def test_record_is_a_noop_when_symbol_missing(monkeypatch, tmp_path):
    log_file = tmp_path / "log.json"
    monkeypatch.setattr(performance_log.config, "COMPANY_PERFORMANCE_LOG_FILE", log_file)

    performance_log.record("v1", "", "Some Company")

    assert not log_file.exists()


def test_record_never_raises_on_write_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(performance_log.config, "COMPANY_PERFORMANCE_LOG_FILE", tmp_path / "nosuchdir" / "log.json")
    performance_log.record("v1", "CDSL", "CDSL Ltd")  # must not raise
