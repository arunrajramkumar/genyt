import json

from scripts import merge_usage_counter


def _write(path, data):
    path.write_text(json.dumps(data))


def _run_merge(monkeypatch, base, ours, theirs):
    monkeypatch.setattr(
        "sys.argv", ["merge_usage_counter.py", str(base), str(ours), str(theirs)]
    )
    return merge_usage_counter.main()


def test_sums_independent_increments_from_common_base(tmp_path, monkeypatch):
    base, ours, theirs = tmp_path / "base.json", tmp_path / "ours.json", tmp_path / "theirs.json"
    _write(base, {"month": "2026-10", "calls": 9})
    _write(ours, {"month": "2026-10", "calls": 10})
    _write(theirs, {"month": "2026-10", "calls": 12})

    assert _run_merge(monkeypatch, base, ours, theirs) == 0
    assert json.loads(ours.read_text()) == {"month": "2026-10", "calls": 13}


def test_takes_later_month_as_is_on_month_rollover(tmp_path, monkeypatch):
    base, ours, theirs = tmp_path / "base.json", tmp_path / "ours.json", tmp_path / "theirs.json"
    _write(base, {"month": "2026-10", "calls": 390000})
    _write(ours, {"month": "2026-11", "characters": 500})
    _write(theirs, {"month": "2026-10", "characters": 395000})

    assert _run_merge(monkeypatch, base, ours, theirs) == 0
    assert json.loads(ours.read_text()) == {"month": "2026-11", "characters": 500}


def test_missing_base_treated_as_zero(tmp_path, monkeypatch):
    base, ours, theirs = tmp_path / "missing.json", tmp_path / "ours.json", tmp_path / "theirs.json"
    _write(ours, {"month": "2026-10", "calls": 1})
    _write(theirs, {"month": "2026-10", "calls": 2})

    assert _run_merge(monkeypatch, base, ours, theirs) == 0
    assert json.loads(ours.read_text()) == {"month": "2026-10", "calls": 3}
