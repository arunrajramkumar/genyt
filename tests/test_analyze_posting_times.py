from unittest import mock

from scripts import analyze_posting_times


def test_main_reports_nothing_when_no_videos(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["analyze_posting_times.py"])
    monkeypatch.setattr(analyze_posting_times.youtube_insights, "fetch_channel_insights", lambda max_videos: [])
    analyze_posting_times.main()
    assert "Not enough uploaded videos" in capsys.readouterr().out


def test_main_reports_no_pattern_when_no_repeated_hours(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["analyze_posting_times.py"])
    videos = [{"published_at": "2024-01-01T05:00:00Z", "views_per_day": 100}]
    monkeypatch.setattr(analyze_posting_times.youtube_insights, "fetch_channel_insights", lambda max_videos: videos)
    analyze_posting_times.main()
    assert "no UTC hour has been posted to more than once" in capsys.readouterr().out


def test_main_prints_ranked_report_and_recommendation(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["analyze_posting_times.py", "--max-videos", "50"])
    videos = [
        {"published_at": "2024-01-01T05:00:00Z", "views_per_day": 100},
        {"published_at": "2024-02-01T05:00:00Z", "views_per_day": 200},
    ]
    fetch = mock.Mock(return_value=videos)
    monkeypatch.setattr(analyze_posting_times.youtube_insights, "fetch_channel_insights", fetch)
    analyze_posting_times.main()
    fetch.assert_called_once_with(50)
    out = capsys.readouterr().out
    assert "05:00 UTC" in out
    assert "Best-performing hour" in out
