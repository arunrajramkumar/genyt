"""Regression tests for the 2026-10-22 Swan Defence run: a Groq
json_validate_failed (empty generation) crashed the whole run, and a one-off
quarter (net profit Rs 567 crore on Rs 69 crore revenue, "56800% profit
growth") leaked into the hook and the did-you-know facts."""
import json
from unittest import mock

import pytest
import requests

from pipeline import company_insights as ci
from pipeline import config, fundamentals, llm, segmented_script, trending_stocks
from pipeline.segmented_script import parse_segments
from scripts.build_finance_queue import build_theme_entry

REAL_IS_FAITHFUL = ci._is_faithful
REAL_ARTICULATE = ci.articulate_facts

SWAN = {
    "symbol": "SWANDEF",
    "ratios": {"ROCE": "-7.61 %", "ROE": "-128 %", "Stock P/E": "43.0"},
    "quarterly": {"sales_cr": ["40", "50", "55", "60", "69"], "net_profit_cr": ["1", "-5", "-3", "2", "567"]},
    "revenue_growth_yoy_pct": 72.5,
    "profit_growth_yoy_pct": 56600.0,
}


def _resp(status=200, body=None, text=""):
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = body or {}
    r.text = text or json.dumps(body or {})
    r.headers = {}
    r.raise_for_status.side_effect = requests.HTTPError(f"{status}") if status >= 400 else None
    return r


OK = {"choices": [{"message": {"content": '{"ok": true}'}}]}
JSON_FAIL = {"error": {"code": "json_validate_failed", "failed_generation": ""}}


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def install(responses):
        it = iter(responses)

        def post(url, headers, json, timeout):
            calls.append(json)
            return next(it)
        monkeypatch.setattr(llm.requests, "post", post)
        return calls
    return install


# ---- llm.chat_json resilience ----

def test_last_json_retry_drops_json_mode_and_can_succeed(groq):
    calls = groq([_resp(400, JSON_FAIL)] * 3 + [_resp(200, OK)])
    assert llm.chat_json("s", "u", max_tokens=512, timeout=5) == '{"ok": true}'
    assert [c["max_tokens"] for c in calls] == [512, 1024, 2048, 4096]
    assert all("response_format" in c for c in calls[:3]) and "response_format" not in calls[3]


def test_exhausted_json_retries_raise_a_value_error(groq):
    groq([_resp(400, JSON_FAIL)] * 4)
    with pytest.raises(ValueError):
        llm.chat_json("s", "u", max_tokens=512, timeout=5)


def test_rate_limit_then_json_failure_then_success_in_any_order(groq):
    calls = groq([_resp(429, {"error": {"message": "try again in 0.1s"}}), _resp(400, JSON_FAIL),
                  _resp(429, {}), _resp(200, OK)])
    assert llm.chat_json("s", "u", max_tokens=512, timeout=5) == '{"ok": true}'
    assert len(calls) == 4


def test_server_errors_are_retried(groq):
    groq([_resp(503, {}), _resp(200, OK)])
    assert llm.chat_json("s", "u", max_tokens=100, timeout=5) == '{"ok": true}'


def test_persistent_server_error_is_runtime_error(groq):
    groq([_resp(503, {})] * 3)
    with pytest.raises(RuntimeError, match="503"):
        llm.chat_json("s", "u", max_tokens=100, timeout=5)


def test_empty_content_raises_value_error(groq):
    groq([_resp(200, {"choices": [{"message": {"content": ""}}]})])
    with pytest.raises(llm.LLMResponseError):
        llm.chat_json("s", "u", max_tokens=100, timeout=5)


def test_caller_budget_above_cap_is_never_lowered(groq):
    calls = groq([_resp(400, JSON_FAIL), _resp(200, OK)])
    llm.chat_json("s", "u", max_tokens=6000, timeout=5)
    assert [c["max_tokens"] for c in calls] == [6000, 6000]


# ---- callers resample instead of crashing ----

def test_segment_scene_survives_an_llm_response_error(monkeypatch):
    replies = iter([llm.LLMResponseError("empty"), json.dumps({"narration": "Profit jumped.", "visual_query": "x"})])

    def chat(*a, **k):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(segmented_script.llm, "chat_json", chat)
    scene = segmented_script._write_segment_scene("Hook", "Profit grew 52%", 6)
    assert scene["narration"] == "Profit jumped."


def test_trending_keeps_other_chunks_when_one_hard_fails(monkeypatch):
    monkeypatch.setattr(trending_stocks, "MAX_EXTRACT_CHARS", 60)
    replies = iter([RuntimeError("Groq API error 413"),
                    json.dumps({"candidates": [{"symbol": "INFY", "name": "Infosys", "event": "other"}]})])

    def chat(*a, **k):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(trending_stocks.llm, "chat_json", chat)
    text = "Tata Consultancy Services posted results today.\nInfosys shares moved on the news today."
    assert [c["name"] for c in trending_stocks._extract_candidates(text)] == ["Infosys"]


def test_trending_ignores_non_object_replies(monkeypatch):
    monkeypatch.setattr(trending_stocks.llm, "chat_json", lambda *a, **k: '["Infosys"]')
    assert trending_stocks._extract_candidates("Infosys shares moved today.") == []


# ---- growth wording ----

@pytest.mark.parametrize("values, noun, expected", [
    (["100", "1", "1", "1", "137.3"], "Net profit", "Net profit grew 37.3% year-on-year"),
    (["100", "1", "1", "1", "91.8"], "Revenue", "Revenue fell 8.2% year-on-year"),
    (["1", "-5", "-3", "2", "567"], "Net profit", "Net profit rose from Rs 1 crore to Rs 567 crore year-on-year"),
    (["-12", "1", "1", "1", "40"], "Net profit", "Net profit went from a loss of Rs 12 crore to a profit of Rs 40 crore year-on-year"),
    (["30", "1", "1", "1", "-8"], "Net profit", "Net profit went from a profit of Rs 30 crore to a loss of Rs 8 crore year-on-year"),
    (["1", "2"], "Revenue", ""),
])
def test_growth_sentence(values, noun, expected):
    assert fundamentals.growth_sentence(values, noun) == expected


def test_plain_growth_pct_rejects_tiny_bases_and_losses():
    assert fundamentals.plain_growth_pct(["100", "1", "1", "1", "152"]) == 52.0
    assert fundamentals.plain_growth_pct(SWAN["quarterly"]["net_profit_cr"]) is None
    assert fundamentals.plain_growth_pct(["-12", "1", "1", "1", "40"]) is None


def test_profit_exceeds_revenue():
    assert fundamentals.profit_exceeds_revenue(SWAN)
    assert not fundamentals.profit_exceeds_revenue({"quarterly": {"sales_cr": ["222"], "net_profit_cr": ["152"]}})
    assert not fundamentals.profit_exceeds_revenue({})


def test_format_fundamentals_facts_flags_one_off_profit():
    text = fundamentals.format_fundamentals_facts(SWAN)
    assert "56600" not in text and "rose from Rs 1 crore to Rs 567 crore" in text
    assert "other income or one-off items" in text


# ---- the Swan Defence theme entry ----

def test_swan_like_entry_hooks_on_revenue_not_absurd_profit_growth(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda s: SWAN)
    entry = build_theme_entry("SWANDEF", "Swan Defence and Heavy Industries Ltd.")
    segments = {s["label"]: s["content"] for s in parse_segments(entry["theme"])}
    assert "Revenue growth: 72.5% year-on-year" in segments["Hook"]
    assert "56600" not in entry["theme"] and "Ltd.." not in entry["theme"]
    growth = segments["Revenue & Profit Growth"]
    assert "rose from Rs 1 crore to Rs 567 crore" in growth and "one-off" in growth


def test_one_off_quarter_suppresses_profit_based_facts():
    assert ci.profit_per_day_fact(SWAN) == ""
    assert ci.margin_fact(SWAN) == ""


def test_margin_over_100_percent_is_skipped():
    data = {"quarterly": {"sales_cr": ["100", "1", "1", "1", "100"], "net_profit_cr": ["20", "1", "1", "1", "100.5"]}}
    assert ci.margin_fact(data) == ""


# ---- faithfulness second opinion ----

def _verdict(monkeypatch, reply):
    def chat(*a, **k):
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(ci.llm, "chat_json", chat)


@pytest.mark.parametrize("reply, expected", [
    ('{"faithful": true, "reason": "ok"}', True),
    ('{"faithful": false, "reason": "upgraded licence to built"}', False),
    ('{"faithful": "yes"}', False),
    ("not json", False),
    (RuntimeError("down"), False),
])
def test_is_faithful_fails_closed(monkeypatch, reply, expected):
    _verdict(monkeypatch, reply)
    assert REAL_IS_FAITHFUL("claim", "source") is expected


def test_articulation_rejected_when_judged_unfaithful(monkeypatch):
    narration = "Swan Defence built India's first private-sector warships, a remarkable milestone for the yard."
    monkeypatch.setattr(ci.llm, "chat_json", lambda *a, **k: json.dumps({"narration": narration}))
    monkeypatch.setattr(ci, "_is_faithful", lambda claim, source: False)
    facts = ["Swan Defence and Heavy Industries was the first private sector company in India to obtain a "
             "license and contract to build warships."]
    assert REAL_ARTICULATE("Swan Defence and Heavy Industries Ltd", facts) == ""


def test_story_fact_rejected_when_judged_unfaithful_to_its_quote(monkeypatch):
    source = "Pidilite Industries was founded in 1959 and its first product was Fevicol, a white adhesive for carpenters."
    monkeypatch.setattr(ci.llm, "chat_json", lambda *a, **k: json.dumps({
        "fact": "Pidilite Industries invented white adhesive.",
        "quote": "its first product was Fevicol, a white adhesive for carpenters"}))
    monkeypatch.setattr(ci, "_is_faithful", lambda claim, source: False)
    assert ci._extract_story_fact("Pidilite Industries Ltd", source) == ""


# ---- company name extraction (2026-10-23 Hindustan Zinc title bug) ----

HOOK = ("Write an exciting opening question that leads with this headline number -- Profit growth: 140.0% "
        "year-on-year -- in the first 6-8 words, before anything else; never open with the company name, a date, "
        "or other background framing first. Mention Hindustan Zinc (NSE: HINDZINC) only after that.")


def test_mid_sentence_mention_is_trimmed_to_the_name():
    assert segmented_script._extract_company_name(HOOK) == "Hindustan Zinc"
    assert segmented_script._extract_company_name("For Hindustan Zinc (NSE: HINDZINC), pick 3 metrics.") == "Hindustan Zinc"


@pytest.mark.parametrize("prompt, expected", [
    ("Karur Vysya Bank Ltd (NSE: KARURVYSYA) — Q2 profit growth\n0-6 seconds - Hook", "Karur Vysya Bank Ltd"),
    ("Dr. Reddy's Laboratories Ltd. (NSE: DRREDDY)\n...", "Dr. Reddy's Laboratories Ltd."),
    ("## Larsen & Toubro (NSE: LT)\n", "Larsen & Toubro"),
    ("This video covers Dr. Reddy's Laboratories (NSE: DRREDDY) results.", "Dr. Reddy's Laboratories"),
    ("no company here", None),
])
def test_company_name_extraction(prompt, expected):
    assert segmented_script._extract_company_name(prompt) == expected


def test_line_start_mention_wins_over_an_earlier_mid_sentence_one():
    assert segmented_script._extract_company_name(HOOK + "\nHindustan Zinc (NSE: HINDZINC)") == "Hindustan Zinc"


def test_generated_theme_opens_with_clean_company_anchor(monkeypatch):
    monkeypatch.setattr("scripts.build_finance_queue.fundamentals.fetch_fundamentals", lambda s: SWAN)
    theme = build_theme_entry("HINDZINC", "Hindustan Zinc")["theme"]
    assert theme.splitlines()[0] == "Hindustan Zinc (NSE: HINDZINC)"
    assert segmented_script._extract_company_name(theme) == "Hindustan Zinc"
    assert parse_segments(theme)[0]["label"] == "Hook"


# ---- deferred engagement comment ----

from pipeline import youtube_upload  # noqa: E402


def test_engagement_line_round_trips_through_the_description():
    q = "Does the valuation already price in the growth?"
    desc = f"Some summary.\n\n{youtube_upload.engagement_line(q)}\n\n#Shorts"
    assert youtube_upload.question_from_description(desc) == q
    assert youtube_upload.question_from_description("no question line") == ""


def _fake_youtube(videos, threads_by_video, posted):
    yt = mock.Mock()
    yt.channels().list().execute.return_value = {
        "items": [{"id": "UC_me", "contentDetails": {"relatedPlaylists": {"uploads": "UU_me"}}}]}
    yt.playlistItems().list().execute.return_value = {
        "items": [{"contentDetails": {"videoId": v["id"]}} for v in videos]}
    yt.videos().list().execute.return_value = {"items": videos}

    def threads_list(part, videoId, maxResults):
        m = mock.Mock()
        m.execute.return_value = {"items": threads_by_video.get(videoId, [])}
        return m
    yt.commentThreads().list.side_effect = threads_list
    return yt


def _video(vid, privacy, question):
    desc = f"Summary.\n\n{youtube_upload.engagement_line(question)}" if question else "Summary."
    return {"id": vid, "status": {"privacyStatus": privacy}, "snippet": {"description": desc}}


def test_pending_comments_only_for_public_videos_without_our_comment(monkeypatch):
    mine = [{"snippet": {"topLevelComment": {"snippet": {"authorChannelId": {"value": "UC_me"}}}}}]
    other = [{"snippet": {"topLevelComment": {"snippet": {"authorChannelId": {"value": "UC_viewer"}}}}}]
    videos = [_video("pub_new", "public", "Is growth priced in?"),
              _video("pub_done", "public", "Is growth priced in?"),
              _video("priv", "private", "Is growth priced in?"),
              _video("pub_noq", "public", "")]
    posted = []
    monkeypatch.setattr(youtube_upload, "_client", lambda: _fake_youtube(videos, {"pub_new": other, "pub_done": mine}, posted))
    monkeypatch.setattr(youtube_upload, "post_engagement_comment", lambda vid, text: posted.append((vid, text)))
    assert youtube_upload.post_pending_engagement_comments(on_progress=lambda m: None) == 1
    assert posted == [("pub_new", "Is growth priced in? Reply YES or NO below 👇")]


def test_one_failing_video_does_not_stop_the_rest(monkeypatch):
    videos = [_video("a", "public", "Q one?"), _video("b", "public", "Q two?")]
    posted = []

    def post(vid, text):
        if vid == "a":
            raise RuntimeError("commentsDisabled")
        posted.append(vid)
    monkeypatch.setattr(youtube_upload, "_client", lambda: _fake_youtube(videos, {}, posted))
    monkeypatch.setattr(youtube_upload, "post_engagement_comment", post)
    assert youtube_upload.post_pending_engagement_comments(on_progress=lambda m: None) == 1 and posted == ["b"]
