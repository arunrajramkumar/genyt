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


# -- get_style_guidance: prints *why* it returned empty, so a silently no-op'd
# feedback loop (e.g. a token missing yt-analytics.readonly) is observable in
# CI logs instead of looking identical to "ran fine, nothing to report" -----

def test_get_style_guidance_prints_reason_when_not_configured(monkeypatch, capsys):
    monkeypatch.setattr(
        yi, "fetch_channel_insights",
        mock.Mock(side_effect=youtube_upload.NotConfiguredError("no token")),
    )
    yi.get_style_guidance()
    assert "no token" in capsys.readouterr().out


def test_get_style_guidance_prints_reason_when_too_few_videos(monkeypatch, capsys):
    monkeypatch.setattr(yi, "fetch_channel_insights", lambda max_videos=20: [])
    yi.get_style_guidance()
    assert "youtube_insights" in capsys.readouterr().out


def test_get_style_guidance_prints_reason_on_unexpected_error(monkeypatch, capsys):
    monkeypatch.setattr(yi, "fetch_channel_insights", mock.Mock(side_effect=RuntimeError("boom")))
    yi.get_style_guidance()
    assert "boom" in capsys.readouterr().out


def test_get_style_guidance_prints_success_when_guidance_found(monkeypatch, capsys):
    videos = [{"title": "T", "tags": ["a"], "views": 100, "ctr": 5.0, "avg_view_duration_sec": 20.0}]
    monkeypatch.setattr(yi, "fetch_channel_insights", lambda max_videos=20: videos)
    monkeypatch.setattr(yi, "summarize_insights", lambda v: "- Lead with a number")
    yi.get_style_guidance()
    out = capsys.readouterr().out
    assert "[youtube_insights]" in out
    assert "1 video" in out


# -- get_combined_guidance: merges own-channel + niche-wide guidance ----------

def test_get_combined_guidance_merges_both_sources(monkeypatch):
    monkeypatch.setattr(yi, "get_style_guidance", lambda: "- Own channel bullet")
    monkeypatch.setattr(yi.market_trends, "get_market_trend_guidance", lambda: "- Niche bullet")
    combined = yi.get_combined_guidance()
    assert "Own channel bullet" in combined
    assert "Niche bullet" in combined


def test_get_combined_guidance_prints_the_guidance_actually_used(monkeypatch, capsys):
    monkeypatch.setattr(yi, "get_style_guidance", lambda: "- Own channel bullet")
    monkeypatch.setattr(yi.market_trends, "get_market_trend_guidance", lambda: "- Niche bullet")
    yi.get_combined_guidance()
    out = capsys.readouterr().out
    assert "Own channel bullet" in out
    assert "Niche bullet" in out


def test_get_combined_guidance_omits_empty_sections(monkeypatch):
    monkeypatch.setattr(yi, "get_style_guidance", lambda: "")
    monkeypatch.setattr(yi.market_trends, "get_market_trend_guidance", lambda: "- Niche bullet")
    combined = yi.get_combined_guidance()
    assert combined == "Broader finance/stock-analysis/wealth niche trends right now:\n- Niche bullet"


def test_get_combined_guidance_returns_empty_when_both_sources_empty(monkeypatch):
    monkeypatch.setattr(yi, "get_style_guidance", lambda: "")
    monkeypatch.setattr(yi.market_trends, "get_market_trend_guidance", lambda: "")
    assert yi.get_combined_guidance() == ""


def test_get_combined_guidance_prints_when_both_sources_empty(monkeypatch, capsys):
    monkeypatch.setattr(yi, "get_style_guidance", lambda: "")
    monkeypatch.setattr(yi.market_trends, "get_market_trend_guidance", lambda: "")
    yi.get_combined_guidance()
    assert "no channel or niche guidance" in capsys.readouterr().out


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
        "v1": {"title": "First video", "tags": ["a", "b"], "publishedAt": "2024-01-01T00:00:00Z"},
        "v2": {"title": "Second video", "tags": ["c"], "publishedAt": "2024-01-01T00:00:00Z"},
        "v3": {"title": "Third video", "tags": []},
    })
    monkeypatch.setattr(yi, "_analytics_rows", lambda analytics, ids: {
        "v1": {"views": 1000, "impressionsClickThroughRate": 6.5, "averageViewDuration": 15.0},
    })

    videos = yi.fetch_channel_insights()

    assert videos[0]["title"] == "First video"
    assert videos[0]["views"] == 1000
    assert videos[0]["ctr"] == 6.5
    assert videos[0]["avg_view_duration_sec"] == 15.0
    assert videos[0]["views_per_day"] > 0
    # Videos with no matching analytics row still show up, with sensible defaults.
    assert videos[1]["title"] == "Second video"
    assert videos[1]["views"] == 0
    assert videos[1]["ctr"] is None
    assert videos[1]["views_per_day"] == 0.0
    # A video with no publishedAt at all must not raise/crash the normalization.
    assert videos[2]["views_per_day"] == 0.0


# -- _analytics_rows: falls back when the full metric set isn't queryable -----

def test_analytics_rows_queries_a_valid_end_date():
    """endDate must be a real, already-elapsed date -- the Analytics API
    rejects a future end date outright (e.g. the old hardcoded 2100-01-01),
    which silently zeroed out all channel-performance guidance."""
    from datetime import date

    analytics = mock.Mock()
    analytics.reports.return_value.query.return_value.execute.return_value = {
        "columnHeaders": [{"name": "video"}, {"name": "views"}],
        "rows": [],
    }

    yi._analytics_rows(analytics, ["v1"])

    _, kwargs = analytics.reports.return_value.query.call_args
    assert kwargs["endDate"] == date.today().isoformat()


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


# -- _views_per_day: normalizes raw view count by days since upload ----------

def test_views_per_day_returns_zero_for_missing_published_at():
    assert yi._views_per_day(1000, None) == 0.0
    assert yi._views_per_day(1000, "") == 0.0


def test_views_per_day_returns_zero_for_unparseable_published_at():
    assert yi._views_per_day(1000, "not-a-date") == 0.0


def test_views_per_day_floors_at_one_day_for_a_video_published_today():
    from datetime import date
    published_today = f"{date.today().isoformat()}T00:00:00Z"
    assert yi._views_per_day(500, published_today) == 500.0


def test_views_per_day_divides_by_days_live():
    from datetime import date, timedelta
    published_10_days_ago = f"{(date.today() - timedelta(days=10)).isoformat()}T00:00:00Z"
    assert yi._views_per_day(1000, published_10_days_ago) == 100.0


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


def test_summarize_insights_handles_guidance_returned_as_a_list(monkeypatch):
    """Regression test: a production run had the model return "guidance" as a
    JSON array of bullets instead of the single string the prompt asks for --
    `.strip()` on a list raised AttributeError and silently killed the whole
    guidance call."""
    monkeypatch.setattr(
        yi.llm, "chat_json",
        lambda *a, **k: _mock_response({"guidance": ["- Lead with a number", "- Keep hooks under 8 words"]}),
    )
    videos = [{"title": "T", "tags": ["a"], "views": 10, "ctr": None, "avg_view_duration_sec": None}]
    assert yi.summarize_insights(videos) == "- Lead with a number\n- Keep hooks under 8 words"

