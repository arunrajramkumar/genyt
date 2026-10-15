from unittest import mock

from pipeline import market_trends as mt
from pipeline import youtube_upload


def _mock_response(payload: dict) -> str:
    import json
    return json.dumps(payload)


# -- fetch_trending_niche_videos: dedupe + view-count enrichment -------------

def test_fetch_trending_niche_videos_dedupes_across_queries_and_sorts_by_views(monkeypatch):
    monkeypatch.setattr(youtube_upload, "_load_credentials", lambda: mock.Mock())
    monkeypatch.setattr(mt, "build", lambda *a, **k: mock.Mock())

    def fake_search(youtube, query, published_after):
        if query == "stock analysis":
            return [{"video_id": "v1", "title": "Stock pick A", "channel_title": "ChanA"}]
        return [{"video_id": "v1", "title": "Stock pick A", "channel_title": "ChanA"},
                {"video_id": "v2", "title": "Wealth tip B", "channel_title": "ChanB"}]

    monkeypatch.setattr(mt, "_search_videos", fake_search)
    monkeypatch.setattr(mt, "_video_view_counts", lambda youtube, ids: {"v1": 500, "v2": 9000})

    videos = mt.fetch_trending_niche_videos()

    assert [v["video_id"] for v in videos] == ["v2", "v1"]  # sorted by views, deduped
    assert videos[0]["views"] == 9000


def test_fetch_trending_niche_videos_returns_empty_when_no_results(monkeypatch):
    monkeypatch.setattr(youtube_upload, "_load_credentials", lambda: mock.Mock())
    monkeypatch.setattr(mt, "build", lambda *a, **k: mock.Mock())
    monkeypatch.setattr(mt, "_search_videos", lambda youtube, query, published_after: [])
    assert mt.fetch_trending_niche_videos() == []


# -- summarize_market_trends: LLM call shape ----------------------------------

def test_summarize_market_trends_returns_empty_for_no_videos():
    assert mt.summarize_market_trends([]) == ""


def test_summarize_market_trends_extracts_guidance_from_model_response(monkeypatch):
    monkeypatch.setattr(
        mt.llm, "chat_json",
        lambda *a, **k: _mock_response({"guidance": "- Name a specific stock in the title"}),
    )
    videos = [{"title": "T", "channel_title": "C", "views": 1000}]
    assert mt.summarize_market_trends(videos) == "- Name a specific stock in the title"


def test_summarize_market_trends_returns_empty_after_bad_responses(monkeypatch):
    monkeypatch.setattr(mt.llm, "chat_json", lambda *a, **k: "not json at all")
    videos = [{"title": "T", "channel_title": "C", "views": 1}]
    assert mt.summarize_market_trends(videos) == ""


def test_summarize_market_trends_handles_guidance_returned_as_a_list(monkeypatch):
    """Regression test: same production failure as youtube_insights -- the
    model returned "guidance" as a JSON array instead of a single string."""
    monkeypatch.setattr(
        mt.llm, "chat_json",
        lambda *a, **k: _mock_response({"guidance": ["- Name a specific stock", "- Use a bold number"]}),
    )
    videos = [{"title": "T", "channel_title": "C", "views": 1000}]
    assert mt.summarize_market_trends(videos) == "- Name a specific stock\n- Use a bold number"


# -- get_market_trend_guidance: best-effort, never raises ---------------------

def test_get_market_trend_guidance_returns_empty_on_fetch_failure(monkeypatch):
    monkeypatch.setattr(mt, "fetch_trending_niche_videos", mock.Mock(side_effect=RuntimeError("quota exceeded")))
    assert mt.get_market_trend_guidance() == ""


def test_get_market_trend_guidance_returns_summary_when_videos_available(monkeypatch):
    videos = [{"title": "T", "channel_title": "C", "views": 100}]
    monkeypatch.setattr(mt, "fetch_trending_niche_videos", lambda queries=None: videos)
    monkeypatch.setattr(mt, "summarize_market_trends", lambda v: "- Trend bullet")
    assert mt.get_market_trend_guidance() == "- Trend bullet"


def test_get_market_trend_guidance_prints_success_when_guidance_found(monkeypatch, capsys):
    videos = [{"title": "T", "channel_title": "C", "views": 100}]
    monkeypatch.setattr(mt, "fetch_trending_niche_videos", lambda queries=None: videos)
    monkeypatch.setattr(mt, "summarize_market_trends", lambda v: "- Trend bullet")
    mt.get_market_trend_guidance()
    out = capsys.readouterr().out
    assert "[market_trends]" in out
    assert "1 video" in out


# -- get_market_trend_guidance: prints why it returned empty, for the same
# observability reason as youtube_insights.get_style_guidance -----------------

def test_get_market_trend_guidance_prints_reason_on_fetch_failure(monkeypatch, capsys):
    monkeypatch.setattr(mt, "fetch_trending_niche_videos", mock.Mock(side_effect=RuntimeError("quota exceeded")))
    mt.get_market_trend_guidance()
    assert "quota exceeded" in capsys.readouterr().out


def test_get_market_trend_guidance_prints_reason_when_no_videos_found(monkeypatch, capsys):
    monkeypatch.setattr(mt, "fetch_trending_niche_videos", lambda queries=None: [])
    mt.get_market_trend_guidance()
    assert "market_trends" in capsys.readouterr().out

