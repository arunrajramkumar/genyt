from unittest import mock

import pytest

from pipeline import youtube_insights as yi
from pipeline import youtube_upload


def _mock_response(payload: dict) -> str:
    import json
    return json.dumps(payload)


# -- get_style_guidance: never raises, always safe to call -------------------

def test_get_style_guidance_returns_empty_when_not_configured(monkeypatch):
    monkeypatch.setattr(
        youtube_upload, "_load_credentials",
        mock.Mock(side_effect=youtube_upload.NotConfiguredError("nope")),
    )
    assert yi.get_style_guidance() == ""


def test_get_style_guidance_returns_empty_when_too_few_videos(monkeypatch):
    monkeypatch.setattr(yi, "fetch_channel_insights", lambda max_videos=20: [])
    assert yi.get_style_guidance() == ""


def test_get_style_guidance_returns_empty_on_unexpected_error(monkeypatch):
    monkeypatch.setattr(yi, "fetch_channel_insights", mock.Mock(side_effect=RuntimeError("boom")))
    assert yi.get_style_guidance() == ""


def test_get_style_guidance_returns_summary_when_videos_available(monkeypatch):
    videos = [{"title": "T", "tags": ["a"], "views": 100, "ctr": 5.0, "avg_view_duration_sec": 20.0}]
    monkeypatch.setattr(yi, "fetch_channel_insights", lambda max_videos=20: videos)
    monkeypatch.setattr(yi, "summarize_insights", lambda v: "- Lead with a number")
    assert yi.get_style_guidance() == "- Lead with a number"


# -- fetch_channel_insights: shapes the Data API + Analytics API responses ----

def test_fetch_channel_insights_returns_empty_below_minimum_video_count(monkeypatch):
    monkeypatch.setattr(youtube_upload, "_load_credentials", lambda: mock.Mock())
    monkeypatch.setattr(yi, "build", lambda *a, **k: mock.Mock())
    monkeypatch.setattr(yi, "_uploaded_video_ids", lambda youtube, max_videos: ["v1", "v2"])
    assert yi.fetch_channel_insights() == []


def test_fetch_channel_insights_combines_snippet_and_analytics_data(monkeypatch):
    monkeypatch.setattr(youtube_upload, "_load_credentials", lambda: mock.Mock())
    monkeypatch.setattr(yi, "build", lambda *a, **k: mock.Mock())
    monkeypatch.setattr(yi, "_uploaded_video_ids", lambda youtube, max_videos: ["v1", "v2", "v3"])
    monkeypatch.setattr(yi, "_video_snippets", lambda youtube, ids: {
        "v1": {"title": "First video", "tags": ["a", "b"]},
        "v2": {"title": "Second video", "tags": ["c"]},
        "v3": {"title": "Third video", "tags": []},
    })
    monkeypatch.setattr(yi, "_analytics_rows", lambda analytics, ids: {
        "v1": {"views": 1000, "impressionsClickThroughRate": 6.5, "averageViewDuration": 15.0},
    })

    videos = yi.fetch_channel_insights()

    assert videos[0] == {
        "title": "First video", "tags": ["a", "b"], "views": 1000,
        "ctr": 6.5, "avg_view_duration_sec": 15.0,
    }
    # Videos with no matching analytics row still show up, with sensible defaults.
    assert videos[1]["title"] == "Second video"
    assert videos[1]["views"] == 0
    assert videos[1]["ctr"] is None


# -- _analytics_rows: falls back when the full metric set isn't queryable -----

def test_analytics_rows_falls_back_to_basic_metrics_on_failure():
    analytics = mock.Mock()
    full_query = mock.Mock(execute=mock.Mock(side_effect=Exception("403 insufficient access")))
    basic_response = {
        "columnHeaders": [{"name": "video"}, {"name": "views"}],
        "rows": [["v1", 42]],
    }
    basic_query = mock.Mock(execute=mock.Mock(return_value=basic_response))
    analytics.reports.return_value.query.side_effect = [full_query, basic_query]

    rows = yi._analytics_rows(analytics, ["v1"])

    assert rows == {"v1": {"video": "v1", "views": 42}}


def test_analytics_rows_returns_empty_when_both_attempts_fail():
    analytics = mock.Mock()
    analytics.reports.return_value.query.return_value.execute.side_effect = Exception("boom")
    assert yi._analytics_rows(analytics, ["v1"]) == {}


# -- summarize_insights: LLM call shape ---------------------------------------

def test_summarize_insights_returns_empty_for_no_videos():
    assert yi.summarize_insights([]) == ""


def test_summarize_insights_extracts_guidance_from_model_response(monkeypatch):
    monkeypatch.setattr(
        yi.llm, "chat_json",
        lambda *a, **k: _mock_response({"guidance": "- Use a specific number in the title"}),
    )
    videos = [{"title": "T", "tags": ["a"], "views": 10, "ctr": None, "avg_view_duration_sec": None}]
    assert yi.summarize_insights(videos) == "- Use a specific number in the title"


def test_summarize_insights_returns_empty_after_bad_responses(monkeypatch):
    monkeypatch.setattr(yi.llm, "chat_json", lambda *a, **k: "not json at all")
    videos = [{"title": "T", "tags": [], "views": 1, "ctr": None, "avg_view_duration_sec": None}]
    assert yi.summarize_insights(videos) == ""
